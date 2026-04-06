"""
optimization/nom_driver.py

===========================================================================
THE IDEA: CAN WE AUTO-DIFF THROUGH NEURALFOIL?
===========================================================================

SHORT ANSWER: Not truly, but we can get MUCH closer to the professor's
intent than nom_driver.py does.

WHY NEURALFOIL IS NOT DIRECTLY DIFFERENTIABLE:
  NeuralFoil internally calls AeroSandbox, which calls a JAX model that
  is converted to NumPy. The NumPy calls break the TF computation graph.
  Even if we could trace them, NeuralFoil converts coordinates to Kulfan
  parameters via a least-squares solve (lstsq) -- that's not a TF op.

WHAT WE CAN DO INSTEAD (this file):
  Strategy: treat NeuralFoil as a pure function f: R^6 -> R^1 (CD/CL)
  and wrap it with tf.py_function so that GradientTape can track it,
  then register a custom gradient that uses CENTRAL DIFFERENCES.

  This is CLEANER than the old version because:
    - The finite-difference gradient lives in ONE place (the registered
      custom gradient on the py_function wrapper), not inside train_step
    - train_step itself is a plain GradientTape block with no manual
      gradient computation -- it looks like the TF guide example
    - The forward pass is just: loss = neuralfoil_loss(self.z)
    - tape.gradient(loss, [self.z]) calls our FD grad automatically

STRUCTURE:
  NeuralFoilLossOp.__call__(z_tf) -> tf.Tensor (scalar loss)
    |
    +-- @tf.custom_gradient wraps:
         FORWARD:  z -> NF -> CD/CL + penalty  (NumPy, via .numpy())
         BACKWARD: central differences over CD/CL + penalty

  NOMModel.train_step:
    with tf.GradientTape() as tape:
        loss = self.nf_op(self.z)    # all gradient machinery is in nf_op
    grads = tape.gradient(loss, [self.z])
    optimizer.apply_gradients(...)
    -- done. No manual FD in train_step.

CHANGE LOG:
  [3/25/26] Write nom_summary.json at end of run (fixes plotter reading
            stale n_improved / baseline values from old runs).
  [3/25/26] Save foil coords per iteration inside nom_history.json so
            plot_nom_animation.py can render each iter as a frame.
  [3/25/26] Added live_display and save_frames options to nom_optimize()
            for Action Items 1 & 2 (live popup + video export).
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from pipeline.talarai_pipeline import TalarAIPipeline
except ModuleNotFoundError:
    from talarai_pipeline import TalarAIPipeline

try:
    from optimization.objective import default_objective
    from optimization.constraints import latent_minmax_bounds, total_penalty
except ModuleNotFoundError:
    from objective import default_objective
    from constraints import latent_minmax_bounds, total_penalty


# ===========================================================================
# DEFAULT OPERATING POINT
# ===========================================================================
DEFAULT_ALPHA = 2.0
DEFAULT_RE    = 150_000

_VALID_ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
_VALID_RES    = [50_000, 100_000, 150_000, 250_000, 350_000, 450_000]


# ===========================================================================
# HELPERS
# ===========================================================================

def snap_condition(alpha: float, Re: float) -> tuple[float, float]:
    alpha_s = min(_VALID_ALPHAS, key=lambda a: abs(a - alpha))
    Re_s    = min(_VALID_RES,    key=lambda r: abs(r - Re))
    if alpha_s != alpha or Re_s != Re:
        print(f"  [snap] ({alpha}, Re={Re:.0f}) -> ({alpha_s}, Re={Re_s:.0f})")
    return float(alpha_s), float(Re_s)


def load_latent_dataset(csv_path: str = "data/airfoil_latent_params_6.csv") -> np.ndarray:
    df = pd.read_csv(csv_path)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] != 6:
        raise ValueError(f"Expected 6 latent columns, found {numeric.shape[1]}")
    return numeric.values.astype(float)


def load_best_baseline(json_path: str | Path) -> dict | None:
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"  Baseline not found: {json_path}")
        return None
    with open(json_path) as f:
        return json.load(f)


# ===========================================================================
# NEURALFOIL LOSS OPERATOR
# ===========================================================================

class NeuralFoilLossOp:
    """
    Wraps NeuralFoil into a GradientTape-compatible operator.

    The gradient is computed by central finite differences over the 6
    latent dimensions. The FD logic lives HERE, not in train_step.
    """

    def __init__(self, pipeline, alpha: float, Re: float,
                 lat_lo_np, lat_hi_np, penalty_kwargs: dict,
                 cl_min: float, fd_eps: float, bounds_lam: float):
        self._pipeline     = pipeline
        self._alpha        = alpha
        self._Re           = Re
        self._lat_lo_np    = np.array(lat_lo_np, dtype=np.float64)
        self._lat_hi_np    = np.array(lat_hi_np, dtype=np.float64)
        self._penalty_kw   = penalty_kwargs
        self._cl_min       = cl_min
        self._fd_eps       = fd_eps
        self._bounds_lam   = bounds_lam

        # Store last forward info for train_step to read without re-evaluating
        self.last_info: dict | None = None

    def _evaluate(self, z: np.ndarray) -> tuple[float, dict | None]:
        """
        Evaluate NeuralFoil + constraints at latent vector z.
        Returns (loss, info_dict) where loss=1e9 means invalid.
        """
        try:
            out = self._pipeline.eval_latent_with_neuralfoil(
                z, alpha=self._alpha, Re=self._Re)
            CL     = float(out["CL"])
            CD     = float(out["CD"])
            coords = out.get("coords")
        except Exception:
            return 1e9, None

        if not (np.isfinite(CL) and np.isfinite(CD) and CD > 0):
            return 1e9, None
        if CL < self._cl_min:
            return 1e9, None

        obj = default_objective(CL, CD)
        if not np.isfinite(obj):
            return 1e9, None

        pen = 0.0
        if coords is not None:
            try:
                pen, _ = total_penalty(
                    latent_vec=z, coords=coords, CL=CL,
                    lat_lo=self._lat_lo_np, lat_hi=self._lat_hi_np,
                    **self._penalty_kw)
            except Exception:
                pass

        bp = self._bounds_lam * float(np.sum(
            np.maximum(0.0, self._lat_lo_np - z) +
            np.maximum(0.0, z - self._lat_hi_np)))

        loss = float(obj + pen + bp)
        info = {"CL": CL, "CD": CD, "obj": obj, "pen": pen, "bp": bp, "coords": coords}
        return loss, info

    def __call__(self, z_tf: tf.Tensor) -> tf.Tensor:
        """
        Main entry point for GradientTape.
        Returns a scalar tf.Tensor with a registered central-difference gradient.
        """
        op = self

        @tf.custom_gradient
        def _wrapped_call(z_in):
            # FORWARD
            z_np = z_in.numpy().astype(np.float64)
            loss_val, info = op._evaluate(z_np)
            op.last_info   = info

            # BACKWARD: central finite differences
            def grad_fn(upstream, variables=None):
                grad_z = np.zeros(6, dtype=np.float64)
                eps    = op._fd_eps

                for i in range(6):
                    zp = z_np.copy(); zp[i] += eps
                    zm = z_np.copy(); zm[i] -= eps

                    lp, _ = op._evaluate(zp)
                    lm, _ = op._evaluate(zm)

                    if np.isfinite(lp) and np.isfinite(lm):
                        grad_z[i] = (lp - lm) / (2.0 * eps)
                    elif np.isfinite(lp) and np.isfinite(loss_val):
                        grad_z[i] = (lp - loss_val) / eps
                    elif np.isfinite(lm) and np.isfinite(loss_val):
                        grad_z[i] = (loss_val - lm) / eps

                input_grad = upstream * tf.constant(grad_z.astype(np.float32))
                var_grads  = [None] * len(variables or [])
                return input_grad, var_grads

            return tf.constant(float(loss_val), dtype=tf.float32), grad_fn

        return _wrapped_call(z_tf)


# ===========================================================================
# NOM MODEL
# ===========================================================================

class NOMModel(tf.keras.Model):
    """
    NOM optimizer using GradientTape + NeuralFoilLossOp.

    TRAINABLE:  z[6]
    FROZEN:     decoder (held by pipeline, not by this model)

    Live display and frame-saving are opt-in via setup_live_display().
    """

    def __init__(self, nf_op: NeuralFoilLossOp, z_init: np.ndarray,
                 pipeline, alpha, Re, lat_lo_np, lat_hi_np, cl_min):
        super().__init__()

        self.nf_op = nf_op

        z0 = np.array(z_init, dtype=np.float32).reshape(6)
        self.z = self.add_weight(
            name="z",
            shape=(6,),
            initializer=tf.constant_initializer(z0),
            trainable=True,
        )

        self._pipeline   = pipeline
        self._alpha      = float(alpha)
        self._Re         = float(Re)
        self._cl_min     = float(cl_min)
        self._lat_lo_np  = lat_lo_np
        self._lat_hi_np  = lat_hi_np

        # Optimization tracking
        self.best_result   = None
        self.best_loss     = float("inf")
        self.history_log   = []       # list of per-iter dicts (includes coords)
        self.n_valid       = 0
        self.n_skipped     = 0
        self.n_improved    = 0
        self._n_iters      = 500
        self._t_start      = None
        self._last_improved = False

        # Live display state (set up via setup_live_display())
        self._live_fig       = None
        self._live_ax_foil   = None
        self._live_ax_conv   = None
        self._baseline_coords = None
        self._baseline_label  = "baseline"
        self._save_frames     = False
        self._frames_dir      = None
        self._objs_live       = []
        self._best_live       = []

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    # ------------------------------------------------------------------
    # LIVE DISPLAY SETUP  (Action Items 1 & 2)
    # ------------------------------------------------------------------

    def setup_live_display(self,
                           baseline_coords: np.ndarray | None,
                           baseline_label: str = "baseline",
                           frames_dir: str | Path | None = None):
        """
        Call before nom.fit() to enable a live matplotlib popup that updates
        every iteration, showing the foil shape and convergence curve.

        If frames_dir is given, each frame is also saved as a PNG there.
        After training, run:
            python plot_nom_animation.py  (to get the interactive slider)
            ffmpeg -r 30 -i outputs/frames/frame_%04d.png -vcodec libx264 nom_opt.mp4
        """
        import matplotlib
        matplotlib.use("TkAgg")   # needed for interactive window on Windows
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        self._baseline_coords = baseline_coords
        self._baseline_label  = baseline_label

        if frames_dir is not None:
            self._save_frames = True
            self._frames_dir  = Path(frames_dir)
            self._frames_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Frames will be saved to: {self._frames_dir}")

        plt.ion()
        fig = plt.figure(figsize=(12, 5))
        fig.patch.set_facecolor("#1a1a2e")
        gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

        ax_foil = fig.add_subplot(gs[0])
        ax_conv = fig.add_subplot(gs[1])

        for ax in [ax_foil, ax_conv]:
            ax.set_facecolor("#0f0f23")
            ax.tick_params(colors="#aaaacc", labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor("#333366")
            ax.grid(True, color="#1f1f44", linewidth=0.5, linestyle="--")

        ax_foil.set_xlim(-0.02, 1.05)
        ax_foil.set_ylim(-0.25, 0.25)
        ax_foil.set_aspect("equal")
        ax_foil.set_title("Foil shape (live)", color="#c5cae9", fontsize=10)
        ax_foil.set_xlabel("x/c", color="#aaaacc", fontsize=8)
        ax_foil.set_ylabel("y/c", color="#aaaacc", fontsize=8)

        ax_conv.set_title("CD/CL convergence", color="#c5cae9", fontsize=10)
        ax_conv.set_xlabel("Iteration", color="#aaaacc", fontsize=8)
        ax_conv.set_ylabel("CD / CL  (minimize)", color="#aaaacc", fontsize=8)

        fig.suptitle("TalarAI NOM  —  optimizing live…",
                     color="#c5cae9", fontsize=11, y=1.01)

        self._live_fig     = fig
        self._live_ax_foil = ax_foil
        self._live_ax_conv = ax_conv
        plt.tight_layout()
        plt.pause(0.001)

    def _update_live(self, iter_num: int, coords: np.ndarray | None,
                     CL: float, CD: float, loss: float):
        """Redraw the live figure for this iteration."""
        if self._live_fig is None:
            return
        import matplotlib.pyplot as plt

        n_points = 40
        obj = CD / CL if CL > 0 else float("nan")
        self._objs_live.append(obj if np.isfinite(obj) else np.nan)
        cur_best = self.best_loss if np.isfinite(self.best_loss) else np.nan
        self._best_live.append(1.0 / cur_best if np.isfinite(cur_best) and cur_best > 0 else np.nan)

        ax_foil = self._live_ax_foil
        ax_conv = self._live_ax_conv
        ax_foil.cla()
        ax_conv.cla()

        for ax in [ax_foil, ax_conv]:
            ax.set_facecolor("#0f0f23")
            ax.tick_params(colors="#aaaacc", labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor("#333366")
            ax.grid(True, color="#1f1f44", linewidth=0.5, linestyle="--")

        # --- Foil panel ---
        if coords is not None:
            upper = coords[:n_points][::-1]   # TE->LE reversed to LE->TE
            lower = coords[n_points:]
            ax_foil.plot(upper[:, 0], upper[:, 1], color="#4fc3f7", lw=2.0, label="current upper")
            ax_foil.plot(lower[:, 0], lower[:, 1], color="#81d4fa", lw=2.0, ls="--", label="current lower")
            xf  = np.linspace(0, 1, 200)
            yuf = np.interp(xf, upper[:, 0], upper[:, 1])
            ylf = np.interp(xf, lower[:, 0], lower[:, 1])
            ax_foil.fill_between(xf, ylf, yuf, alpha=0.08, color="#4fc3f7")

        if self._baseline_coords is not None:
            bc = self._baseline_coords
            bu = bc[:n_points][::-1]
            bl = bc[n_points:]
            ax_foil.plot(bu[:, 0], bu[:, 1], color="#ff7043", lw=1.2, ls=":", alpha=0.7,
                         label=self._baseline_label)
            ax_foil.plot(bl[:, 0], bl[:, 1], color="#ff7043", lw=1.2, ls=":", alpha=0.7)

        ld = CL / CD if CD > 0 else 0.0
        best_ld = 1.0 / max(self.best_loss, 1e-9)
        ax_foil.set_xlim(-0.02, 1.05)
        ax_foil.set_ylim(-0.25, 0.25)
        ax_foil.set_aspect("equal")
        ax_foil.set_title(
            f"iter {iter_num}/{self._n_iters}  CL={CL:.4f}  CD={CD:.6f}\n"
            f"L/D={ld:.1f}   best L/D={best_ld:.1f}",
            color="#c5cae9", fontsize=8)
        ax_foil.set_xlabel("x/c", color="#aaaacc", fontsize=8)
        ax_foil.set_ylabel("y/c", color="#aaaacc", fontsize=8)
        ax_foil.legend(fontsize=6, framealpha=0.3, facecolor="#0f0f23",
                       edgecolor="#333366", labelcolor="#c5cae9")

        # --- Convergence panel ---
        iters = list(range(1, len(self._best_live) + 1))
        finite_objs = [o if np.isfinite(o) else np.nan for o in self._objs_live]
        ax_conv.plot(iters, finite_objs, color="#4fc3f7", alpha=0.25, lw=0.6, label="CD/CL")
        ax_conv.plot(iters, self._best_live, color="#ffd54f", lw=2.0, label="best L/D")
        ax_conv.set_xlabel("Iteration", color="#aaaacc", fontsize=8)
        ax_conv.set_ylabel("L/D best  /  CD/CL", color="#aaaacc", fontsize=8)
        ax_conv.legend(fontsize=6, framealpha=0.3, facecolor="#0f0f23",
                       edgecolor="#333366", labelcolor="#c5cae9")

        self._live_fig.suptitle(
            f"TalarAI NOM  —  iter {iter_num}/{self._n_iters}  "
            f"{'★ BEST' if self._last_improved else ''}",
            color="#c5cae9" if not self._last_improved else "#ffd54f",
            fontsize=11, y=1.01)

        self._live_fig.canvas.draw()
        self._live_fig.canvas.flush_events()

        # Save frame PNG if requested
        if self._save_frames and self._frames_dir is not None:
            frame_path = self._frames_dir / f"frame_{iter_num:04d}.png"
            self._live_fig.savefig(frame_path, dpi=100, bbox_inches="tight",
                                   facecolor=self._live_fig.get_facecolor())

    # ------------------------------------------------------------------
    # KERAS PLUMBING
    # ------------------------------------------------------------------

    def call(self, inputs=None, training=False):
        return self.z

    # ------------------------------------------------------------------
    # TRAIN STEP  (clean GradientTape block, matching the TF guide)
    # ------------------------------------------------------------------

    def train_step(self, data):
        if self._t_start is None:
            self._t_start = time.time()

        it       = int(self.optimizer.iterations.numpy())
        iter_num = it + 1

        z_saved = self.z.numpy().copy().astype(np.float64)

        # ---- GRADIENT TAPE ----
        with tf.GradientTape() as tape:
            loss = self.nf_op(self.z)

        fwd_loss = float(loss.numpy())
        fwd_info = self.nf_op.last_info

        # Skip if forward was invalid
        if not np.isfinite(fwd_loss) or fwd_info is None:
            self.z.assign(z_saved.astype(np.float32))
            self.n_skipped += 1
            self._last_improved = False
            self._print_step(iter_num, fwd_loss, {}, skipped=True)
            self.loss_tracker.update_state(1e9)
            return {m.name: m.result() for m in self.metrics}

        # Apply gradients
        gradients = tape.gradient(loss, [self.z])
        self.optimizer.apply_gradients(zip(gradients, [self.z]))

        # Clip z to latent bounds
        z_new = np.clip(
            self.z.numpy().astype(np.float64),
            self._lat_lo_np,
            self._lat_hi_np,
        )
        self.z.assign(z_new.astype(np.float32))

        # Evaluate post-step z for best tracking
        new_loss, new_info = self.nf_op._evaluate(z_new)
        step_ok = np.isfinite(new_loss) and new_info is not None

        # Hard-reject if geometry penalty is triggered
        if step_ok and new_info.get("pen", 0.0) >= 1000.0:
            self.z.assign(z_saved.astype(np.float32))
            step_ok = False
            self.n_skipped += 1
            self._last_improved = False
        elif step_ok:
            self.n_valid += 1
            genuinely_valid = (new_info.get("pen", 0.0) < 1.0) and np.isfinite(new_loss)

            if genuinely_valid and new_loss < self.best_loss:
                self.best_loss = new_loss
                self.best_result = {
                    "latent": z_new.copy(),
                    "coords": new_info.get("coords"),
                    "CL":     new_info["CL"],
                    "CD":     new_info["CD"],
                    "cd_cl":  new_info["obj"],
                }
                self.n_improved += 1
                self._last_improved = True
            else:
                self._last_improved = False

            # -------------------------------------------------------
            # LOG THIS ITERATION
            # Includes coords (as nested list) so plot_nom_animation.py
            # can reconstruct the foil shape at every iteration.
            # -------------------------------------------------------
            coords_arr = new_info.get("coords")
            self.history_log.append({
                "iter":   iter_num,
                "CL":     new_info["CL"],
                "CD":     new_info["CD"],
                "cd_cl":  new_info["obj"],
                "loss":   float(new_loss),
                "pen":    float(new_info.get("pen", 0.0)),
                "coords": coords_arr.tolist() if coords_arr is not None else None,
            })

            # Update live display if enabled
            self._update_live(
                iter_num,
                new_info.get("coords"),
                new_info["CL"],
                new_info["CD"],
                float(new_loss),
            )
        else:
            self.z.assign(z_saved.astype(np.float32))
            self.n_skipped += 1
            self._last_improved = False

        self._print_step(iter_num, fwd_loss, fwd_info, step_ok=step_ok)
        self.loss_tracker.update_state(fwd_loss)
        return {m.name: m.result() for m in self.metrics}

    def _print_step(self, iter_num, loss_0, info,
                    skipped=False, step_ok=True):
        n_total  = getattr(self, "_n_iters", 500)
        elapsed  = time.time() - (self._t_start or time.time())
        secs_per = elapsed / max(iter_num, 1)
        eta      = secs_per * (n_total - iter_num)
        eta_s    = (f"{eta/3600:.1f}h" if eta >= 3600 else
                    f"{eta/60:.0f}m"   if eta >= 60   else
                    f"{eta:.0f}s")

        if skipped or not np.isfinite(loss_0):
            print(f"  iter {iter_num:4d}/{n_total}  SKIP  "
                  f"valid={self.n_valid}  skip={self.n_skipped}  ETA {eta_s}")
            return

        CL  = info.get("CL",  float("nan"))
        CD  = info.get("CD",  float("nan"))
        ld  = CL / CD if (CD and CD > 0) else 0.0
        pen = info.get("pen", 0.0) + info.get("bp", 0.0)
        best_ld = 1.0 / max(self.best_loss, 1e-9)
        status  = "ok" if step_ok else "SKIP"
        star    = " *** BEST" if self._last_improved else ""

        print(f"  iter {iter_num:4d}/{n_total}"
              f"  loss={loss_0:.6f}"
              f"  CL={CL:.4f}  CD={CD:.6f}  L/D={ld:.1f}"
              f"  pen={pen:.5f}"
              f"  best_L/D={best_ld:.1f}"
              f"  {status}"
              f"  valid={self.n_valid}/{iter_num}"
              f"  ETA {eta_s}"
              f"  lr={float(self.optimizer.learning_rate):.1e}"
              f"{star}")


# ===========================================================================
# MAIN OPTIMIZATION FUNCTION
# ===========================================================================

def nom_optimize(
    *,
    alpha: float = DEFAULT_ALPHA,
    Re:    float = DEFAULT_RE,
    n_iters:          int   = 500,
    tf_learning_rate: float = 0.0005,
    fd_eps:     float = 0.01,
    bounds_lam: float = 10.0,
    lam_bounds: float = 1.0,
    lam_geom:   float = 25.0,
    lam_cl:     float = 50.0,
    # THREE-ZONE THICKNESS  (action item 4/6/26 -- see constraints.py for details)
    # Change these numbers once physical manufacturing constraints are known.
    # Current defaults are calibrated to NACA 0012 with ~30-40% headroom:
    #   LE  zone x∈[0.05,0.15]: NACA 0012 has ~7%c here  → floor 2.5%c
    #   Mid zone x∈(0.15,0.75]: NACA 0012 has ~6-12%c    → floor 3.0%c
    #   TE  zone x∈(0.75,0.95]: NACA 0012 has ~1.6-6%c   → floor 0.8%c
    #     (TE zone extended from old 0.90 to 0.95 to prevent razor TE)
    min_thickness_le:  float = 0.025,
    min_thickness_mid: float = 0.030,
    min_thickness_te:  float = 0.008,
    max_thickness:     float = 0.157,
    te_gap_max:        float = 0.005,
    min_max_thickness: float = 0.04,
    # REVISION (4/1/26): set to 0.06 (6%c).
    # MUST match DEFAULT_MAX_CAMBER in build_lookup_table.py -- if these
    # differ, the baseline foil picked by the lookup table may fail the
    # optimizer's own constraint check on iteration 1, causing pen=1000
    # every iteration and valid=0. They are kept in sync at 0.06.
    #
    # Why 0.06: goe803h (~6-7% camber) is a common high-performing baseline.
    # 0.05 was too tight and blocked it. 0.10 was too loose (optimizer hit 8.3%).
    # 0.06 allows the baseline neighborhood while capping exploitation.
    max_camber:        float = 0.06,
    # MINIMUM TE WEDGE ANGLE  (action item 4/6/26 -- see constraints.py for details)
    # NACA 0012 has ~15.5 deg. Floor at 6 deg is very permissive but hard-rejects
    # true knife edges. Raise to 8-10 if the TE is still too sharp after testing.
    min_te_angle_deg:  float = 6.0,
    cl_min: float = 0.15,
    cl_max: float | None = None,
    csv_path:              str       = "data/airfoil_latent_params_6.csv",
    lookup_baseline_path:  str       = "",
    out_path:              str | Path = "outputs",
    # =======================================================================
    # STARTING POINT OPTIONS  (choose ONE — priority order shown below)
    # =======================================================================
    #
    #  OPTION A — foil_name (specific foil from the training database)
    #  ---------------------------------------------------------------
    #  Pass the filename exactly as it appears in airfoils_png/, e.g.:
    #    nom_optimize(foil_name="n0012.png")
    #    nom_optimize(foil_name="e63.png")
    #    nom_optimize(foil_name="goe451.png")
    #
    #  HOW IT WORKS:
    #    Looks up the foil name in airfoil_latent_params_6.csv and grabs
    #    the 6 pre-computed latent parameters for that foil. Those 6 numbers
    #    become z_init — the starting point for the optimizer.
    #
    #  WHY the CSV (not the .dat or .png file):
    #    The optimizer lives entirely in 6D latent space. The decoder already
    #    maps those 6 numbers → foil shape. Using the CSV means we skip
    #    re-running the encoder and use the exact same latent vector that was
    #    used during training. The .dat and .png files are the raw foil data
    #    BEFORE encoding — they're only useful if you want to add a new foil
    #    that isn't in the dataset yet (which would require running encoder.py).
    #
    #  NACA 0012 latent vector (from airfoil_latent_params_6.csv):
    #    n0012.png → [-0.0418, 0.0750, -0.0075, -0.0419, 0.1131, -0.0874]
    #
    #  OPTION B — z_init_csv (resume from a previous run's best result)
    #  ---------------------------------------------------------------
    #  Pass the path to a previous run's best_latent_nom.csv, e.g.:
    #    nom_optimize(z_init_csv="outputs/best_latent_nom.csv")
    #
    #  USE THIS WHEN:
    #    You want to continue optimizing from where a previous run ended,
    #    or re-run the same foil under different constraints (e.g. tighter
    #    camber limit or different alpha/Re condition).
    #
    #  OPTION C — lookup_baseline_path or auto-select (default behavior)
    #  ---------------------------------------------------------------
    #  Leave foil_name="" and z_init_csv="" and the optimizer will:
    #    1) Use lookup_baseline_path if provided, OR
    #    2) Auto-select the best geometry-filtered foil from build_lookup_table
    #       outputs (outputs/best_baseline_foil_alpha{a}_Re{Re}.json)
    #
    #  PRIORITY ORDER (if multiple are set):
    #    z_init_csv > foil_name > lookup_baseline_path > auto-select
    # =======================================================================
    foil_name:  str = "",   # e.g. "n0012.png" — looks up latent in csv_path
    z_init_csv: str = "",   # e.g. "outputs/best_latent_nom.csv" — resume run
    # -----------------------------------------------------------------------
    # Action Item 1: live display + frame saving
    #   live_display=True   -> opens a matplotlib popup window during training
    #   save_frames=True    -> saves PNG frames to outputs/frames/
    #                          (use ffmpeg afterwards to make the video:
    #                           ffmpeg -r 30 -i outputs/frames/frame_%04d.png
    #                                  -vcodec libx264 nom_optimization.mp4)
    # -----------------------------------------------------------------------
    live_display: bool = False,
    save_frames:  bool = False,
):
    """
    Run NOM optimization.

    FD gradient lives in NeuralFoilLossOp; train_step is a clean GradientTape
    block matching the TF guide structure.

    Outputs written to out_path/:
      best_latent_nom.csv      -- best latent vector
      best_latent_nom.npy
      best_coords_nom.csv      -- best foil coordinates (80x2)
      nom_history.json         -- per-iteration log including foil coords
      nom_summary.json         -- summary dict read by plot_nom_results.py
      frames/frame_NNNN.png    -- (if save_frames=True) one PNG per iteration
    """
    alpha, Re = snap_condition(alpha, Re)
    out_path  = Path(out_path)
    out_path.mkdir(exist_ok=True)

    print("=" * 70)
    print("NOM  --  NeuralFoilLossOp + clean GradientTape train_step")
    print("=" * 70)
    print(f"  Iterations:     {n_iters}")
    print(f"  Learning rate:  {tf_learning_rate}")
    print(f"  FD epsilon:     {fd_eps}")
    print(f"  Condition:      alpha={alpha} deg  Re={Re:,.0f}")
    print(f"  Architecture:   FD grad in NeuralFoilLossOp (not in train_step)")
    print(f"  Live display:   {live_display}")
    print(f"  Save frames:    {save_frames}")
    print("=" * 70)
    print()

    all_latents = load_latent_dataset(csv_path)
    lat_lo, lat_hi = latent_minmax_bounds(all_latents)

    pipeline = TalarAIPipeline()
    print(f"  Pipeline ready (decoder: {pipeline.decoder_path.name})\n")

    # ------------------------------------------------------------------
    # RESOLVE STARTING LATENT  (priority: z_init_csv > foil_name > lookup)
    # ------------------------------------------------------------------
    latent_baseline  = None
    baseline_filename = "?"

    # Priority 1: z_init_csv — resume from a previous run's best result
    if z_init_csv:
        p = Path(z_init_csv)
        if p.exists():
            arr = np.loadtxt(str(p), delimiter=",", skiprows=1)
            if arr.ndim > 1:
                arr = arr[0]
            z = arr.astype(float)
            if z.shape[0] == 6:
                latent_baseline   = z
                baseline_filename = f"prev_run:{p.name}"
                print(f"  [start] Loaded previous-run latent: {p.name}")
            else:
                print(f"  WARNING: z_init_csv has {z.shape[0]} values (need 6). Ignored.")
        else:
            print(f"  WARNING: z_init_csv not found: {z_init_csv}. Falling through.")

    # Priority 2: foil_name — look up specific foil in the latent CSV
    if latent_baseline is None and foil_name:
        try:
            df_csv = pd.read_csv(csv_path)
            # Try exact match first, then case-insensitive match without .png
            rows = df_csv[df_csv["filename"] == foil_name]
            if rows.empty:
                needle = foil_name.lower().replace(".png", "")
                rows   = df_csv[
                    df_csv["filename"].str.lower().str.replace(".png","",regex=False) == needle
                ]
            if rows.empty:
                print(f"  WARNING: foil '{foil_name}' not found in {csv_path}.")
                print(f"  Check exact filename in airfoils_png/ (e.g. 'n0012.png').")
                print(f"  Falling back to lookup table baseline.")
            else:
                r = rows.iloc[0]
                latent_baseline   = r[[f"p{i}" for i in range(1, 7)]].values.astype(float)
                baseline_filename = str(r["filename"])
                print(f"  [start] Database foil selected: {baseline_filename}")
                print(f"  z_init: {np.round(latent_baseline, 4)}")
        except Exception as e:
            print(f"  WARNING: foil lookup failed ({e}). Falling back to lookup table.")

    # Priority 3: lookup table baseline (original default behavior)
    if latent_baseline is None:
        if not lookup_baseline_path:
            a_s = min(_VALID_ALPHAS, key=lambda a: abs(a - alpha))
            r_s = min(_VALID_RES,    key=lambda r: abs(r - Re))
            tag = f"alpha{a_s:.1f}_Re{r_s:.1e}"
            lookup_baseline_path = f"outputs/best_baseline_foil_{tag}.json"

        baseline = load_best_baseline(lookup_baseline_path)
        if baseline is None:
            print("  No baseline. Run build_lookup_table.py first.")
            return

        latent_baseline   = np.array(baseline["latent"], dtype=float)
        baseline_filename = baseline.get("filename", "?")
        print(f"  [start] Lookup-table baseline: {baseline_filename}")

    print(f"  z_init:   {np.round(latent_baseline, 4)}\n")

    try:
        bl_out    = pipeline.eval_latent_with_neuralfoil(latent_baseline, alpha=alpha, Re=Re)
        bl_CL     = float(bl_out["CL"])
        bl_CD     = float(bl_out["CD"])
        bl_cd_cl  = default_objective(bl_CL, bl_CD)
        bl_LD     = bl_CL / bl_CD if bl_CD > 0 else 0.0
        bl_coords = bl_out.get("coords")
        print(f"  Baseline L/D = {bl_LD:.1f}  (CL={bl_CL:.4f} CD={bl_CD:.6f})\n")
    except Exception as e:
        print(f"  Baseline eval failed: {e}")
        bl_CL = bl_CD = float("nan")
        bl_cd_cl = float("inf")
        bl_LD = 0.0
        bl_coords = None

    penalty_kwargs = dict(
        lam_bounds=lam_bounds, lam_geom=lam_geom, lam_cl=lam_cl,
        min_thickness_le=min_thickness_le,
        min_thickness_mid=min_thickness_mid,
        min_thickness_te=min_thickness_te,
        max_thickness=max_thickness,
        te_gap_max=te_gap_max, min_max_thickness=min_max_thickness,
        max_camber=max_camber, min_te_angle_deg=min_te_angle_deg,
        cl_min=cl_min, cl_max=cl_max,
    )

    nf_op = NeuralFoilLossOp(
        pipeline=pipeline,
        alpha=alpha, Re=Re,
        lat_lo_np=lat_lo, lat_hi_np=lat_hi,
        penalty_kwargs=penalty_kwargs,
        cl_min=cl_min,
        fd_eps=fd_eps,
        bounds_lam=bounds_lam,
    )

    nom = NOMModel(
        nf_op=nf_op,
        z_init=latent_baseline,
        pipeline=pipeline,
        alpha=alpha, Re=Re,
        lat_lo_np=lat_lo, lat_hi_np=lat_hi,
        cl_min=cl_min,
    )
    nom.build(input_shape=(1, 6))

    nom.best_loss   = bl_cd_cl
    nom.best_result = {
        "latent": latent_baseline.copy(),
        "coords": bl_coords,
        "CL": bl_CL, "CD": bl_CD, "cd_cl": bl_cd_cl,
    }

    # ---- Live display setup ----
    if live_display or save_frames:
        frames_dir = out_path / "frames" if save_frames else None
        nom.setup_live_display(
            baseline_coords=bl_coords,
            baseline_label=baseline_filename,
            frames_dir=frames_dir,
        )
        if save_frames and not live_display:
            # Still need the figure for frame saving even if no interactive window
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend for headless saving

    nom.summary()
    n_train  = sum(int(np.prod(v.shape)) for v in nom.trainable_variables)
    n_frozen = sum(int(np.prod(v.shape)) for v in nom.non_trainable_variables)
    print(f"\n  Trainable:  {n_train} params  (z[6])")
    print(f"  Frozen:     {n_frozen} params")
    print(f"  Gradient:   FD inside NeuralFoilLossOp (hidden from train_step)")
    print(f"  train_step: plain GradientTape, no manual FD\n")

    nom.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf_learning_rate,
            clipnorm=0.5,
        ),
        run_eagerly=True,
    )
    print(f"  NF calls per step: ~{1 + 2*6 + 1} = 1 fwd + 12 FD + 1 post-step\n")

    dummy = (tf.data.Dataset
             .from_tensors(tf.zeros((1, 6), dtype=tf.float32))
             .repeat())

    nom._n_iters = n_iters
    nom._t_start = time.time()
    nom.fit(dummy, epochs=n_iters, steps_per_epoch=1, verbose=0)

    # ---- Save outputs ----
    best = nom.best_result
    if best is None or best.get("coords") is None:
        print("NOM found 0 valid candidates.")
        return

    np.savetxt(out_path / "best_latent_nom.csv",
               best["latent"].reshape(1, -1),
               delimiter=",", header="p1,p2,p3,p4,p5,p6", comments="")
    np.save(out_path / "best_latent_nom.npy", best["latent"])
    np.savetxt(out_path / "best_coords_nom.csv",
               best["coords"], delimiter=",", header="x,y", comments="")

    with open(out_path / "nom_history.json", "w") as f:
        json.dump(nom.history_log, f, indent=2)

    best_LD = best["CL"] / best["CD"] if best["CD"] > 0 else 0.0

    # -----------------------------------------------------------------------
    # FIX: Write nom_summary.json so plot_nom_results.py reads correct values.
    # Previously this file was never written, causing plot_nom_results.py to
    # show stale n_improved / n_valid from a prior run's JSON.
    # -----------------------------------------------------------------------
    summary_data = {
        "baseline_foil_filename":  baseline_filename,
        "baseline_CL":             bl_CL,
        "baseline_CD":             bl_CD,
        "baseline_LD":             bl_LD,
        "best_CL":                 best["CL"],
        "best_CD":                 best["CD"],
        "best_LD":                 best_LD,
        "alpha":                   alpha,
        "Re":                      Re,
        "n_iters":                 n_iters,
        "learning_rate":           tf_learning_rate,
        "fd_eps":                  fd_eps,
        "valid_evals":             nom.n_valid,
        "skipped":                 nom.n_skipped,
        "n_improved":              nom.n_improved,
        "min_thickness_le":        min_thickness_le,
        "min_thickness_mid":       min_thickness_mid,
        "min_thickness_te":        min_thickness_te,
        "max_thickness":           max_thickness,
        "min_max_thickness":       min_max_thickness,
        "max_camber":              max_camber,
        "min_te_angle_deg":        min_te_angle_deg,
        "conditions": [{"alpha": alpha, "Re": Re}],
        # Baseline coords stored so animation can draw it without re-running NF
        "baseline_coords": bl_coords.tolist() if bl_coords is not None else None,
    }
    with open(out_path / "nom_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  Wrote nom_summary.json  (n_improved={nom.n_improved})")

    print("=" * 70)
    print("NOM COMPLETE")
    print("=" * 70)
    print(f"  BASELINE:  L/D={bl_LD:.1f}  CL={bl_CL:.4f}  CD={bl_CD:.6f}")
    print(f"  OPTIMIZED: L/D={best_LD:.1f}  CL={best['CL']:.4f}  CD={best['CD']:.6f}")
    if best_LD > bl_LD and np.isfinite(bl_LD):
        print(f"  IMPROVEMENT: +{(best_LD - bl_LD) / bl_LD * 100:.1f}%")
    print(f"  valid={nom.n_valid}  skipped={nom.n_skipped}  improved={nom.n_improved}")
    print("=" * 70)

    if save_frames:
        frames_dir = out_path / "frames"
        n_frames   = len(list(frames_dir.glob("frame_*.png")))
        print(f"\n  {n_frames} frames saved to {frames_dir}")
        print(f"  To make a 30fps mp4:")
        print(f"    ffmpeg -r 30 -i {frames_dir}/frame_%04d.png "
              f"-vcodec libx264 -pix_fmt yuv420p nom_optimization.mp4")

    if live_display:
        import matplotlib.pyplot as plt
        print("\n  Close the live display window to exit.")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    nom_optimize(foil_name="n0012.png", alpha=2.0, Re=150000)
    # for example:nom_optimize(foil_name="n0012.png", alpha=2.0, Re=150000)