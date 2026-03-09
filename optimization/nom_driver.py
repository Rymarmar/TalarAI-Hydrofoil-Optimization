"""
optimization/nom_driver.py

===========================================================================
MEETING ACTION ITEMS — COMPLETE STATUS
===========================================================================

[2/10] Learning rate <= 1e-3.
    DONE. tf_learning_rate defaults to 5e-4.

[2/17] Lookup table as baseline instead of seeds.
    DONE. Baseline loaded from lookup table JSON.

[2/19] nom_optimize for loop: no training. For training: .fit
    DONE. nom.fit() runs the optimization.

[2/26] ONE unified loop — no separate TF epochs then NOM iterations.
    DONE. nom.fit(epochs=n_iters) = n_iters steps.

[3/5] Take out average conditions — single (alpha, Re).
    DONE. No more loop over conditions. Objective is CD/CL at one point.

[3/5] Turn off def loss_at — restructure.
    DONE. Replaced with _compute_loss() method on NOMModel.

[3/5] Simplify epsilon formula (line 688-689).
    DONE. w_eps_i = fd_eps / (abs(z_init[i]) + 1e-6)

[3/5] USE tf.GradientTape ★ MOST IMPORTANT ★
    DONE. train_step uses tf.custom_gradient + tf.GradientTape.
    Forward pass: runs NeuralFoil (NumPy black box).
    Backward pass: finite differences (professor's whiteboard formula).

===========================================================================
HOW tf.GradientTape WORKS WITH NeuralFoil
===========================================================================

NeuralFoil is a NumPy black box. tf.custom_gradient lets us define
our own backward pass so GradientTape can drive the optimization:

    @tf.custom_gradient
    def forward_pass(w, b):
        loss = run_neuralfoil(w, b)        # forward (black box)
        def grad_fn(dy, variables=None):
            grads = finite_differences()    # backward (our FD code)
            return (dy * grads,), [None for _ in (variables or [])]
        return loss, grad_fn

    with tf.GradientTape() as tape:
        loss = forward_pass(w, b)
    grads = tape.gradient(loss, [w, b])     # calls our grad_fn
    optimizer.apply_gradients(...)           # Adam/SGD updates

Professor's whiteboard (3/5/26):
    df/dx = (f(x0) - f(x0 + D)) / D       # finite differences
    x_n = x0 - a * df/dx                   # gradient descent

===========================================================================
ARCHITECTURE (professor's whiteboard):
===========================================================================

  TRAINABLE:
    z_eff[i] = w[i]*z_init[i] + b[i]     (12 trainable params)

  FROZEN:
    Decoder 6->100->1000->80 -> NeuralFoil -> CL, CD

  OBJECTIVE:
    min CD/CL at single (alpha, Re) + penalty
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
# DEFAULT OPERATING POINT  [3/5/26: single condition, no averaging]
# ===========================================================================
DEFAULT_ALPHA = 2.0
DEFAULT_RE    = 350_000

# Valid grid values for snapping (2/19 action item)
_VALID_ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
_VALID_RES    = [50_000, 100_000, 150_000, 250_000, 350_000, 450_000]


# ===========================================================================
# HELPERS
# ===========================================================================

def snap_condition(alpha: float, Re: float) -> tuple[float, float]:
    """Snap (alpha, Re) to nearest valid grid value."""
    alpha_s = min(_VALID_ALPHAS, key=lambda a: abs(a - alpha))
    Re_s    = min(_VALID_RES,    key=lambda r: abs(r - Re))
    if alpha_s != alpha or Re_s != Re:
        print(f"  [snap] ({alpha}, Re={Re:.0f}) -> ({alpha_s}, Re={Re_s:.0f})")
    return float(alpha_s), float(Re_s)


def load_latent_dataset(csv_path: str = "data/airfoil_latent_params_6.csv") -> np.ndarray:
    """Load 6D latent vectors for all training foils. Used for lat_lo/lat_hi."""
    df = pd.read_csv(csv_path)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] != 6:
        raise ValueError(f"Expected 6 latent columns, found {numeric.shape[1]}")
    return numeric.values.astype(float)


def load_best_baseline(json_path: str | Path) -> dict | None:
    """Load baseline foil from lookup table JSON."""
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"  Baseline not found: {json_path}")
        print(f"  Run: python tools/build_lookup_table.py")
        return None
    with open(json_path) as f:
        return json.load(f)


# ===========================================================================
# LATENT LAYER  (z_eff[i] = w[i] * z_init[i] + b[i])
# ===========================================================================

class LinearLatentLayer(tf.keras.layers.Layer):
    """
    TRAINABLE:     w[6] (scale) + b[6] (shift) = 12 params
    NON-TRAINABLE: z_init[6] (frozen baseline from lookup table)
    w starts at 1.0, b starts at 0.0 -> initially z_eff = z_init.
    """
    def __init__(self, lat_lo, lat_hi, init_latent=None, **kwargs):
        super().__init__(**kwargs)
        z_init = (np.array(init_latent, dtype=np.float32).reshape(6)
                  if init_latent is not None
                  else np.random.uniform(lat_lo, lat_hi).astype(np.float32))
        self._z_init = self.add_weight(
            name="z_init", shape=(6,),
            initializer=tf.constant_initializer(z_init), trainable=False)
        self.w = self.add_weight(
            name="w", shape=(6,), initializer="ones", trainable=True)
        self.b = self.add_weight(
            name="b", shape=(6,), initializer="zeros", trainable=True)

    def call(self, inputs=None):
        z_eff = self.w * self._z_init + self.b
        return tf.expand_dims(z_eff, axis=0)

    def get_effective_latent(self) -> np.ndarray:
        """Public API for external callers."""
        return (self.w.numpy() * self._z_init.numpy() + self.b.numpy()).reshape(6)


# ===========================================================================
# NOM MODEL  (nom.summary -> nom.compile -> nom.fit)
# ===========================================================================

class NOMModel(tf.keras.Model):
    """
    [3/5/26] Uses tf.GradientTape + tf.custom_gradient.

    TRAINABLE:  LinearLatentLayer (12 params: w[6] + b[6])
    FROZEN:     Decoder (6->100->1000->80)
    BLACK BOX:  NeuralFoil (gradients via finite differences)
    """

    def __init__(self, decoder_model, lat_lo, lat_hi, init_latent,
                 pipeline, alpha, Re, lat_lo_np, lat_hi_np,
                 penalty_kwargs, cl_min, fd_eps, bounds_lam):
        super().__init__()
        decoder_model.trainable = False
        self.decoder      = decoder_model
        self.latent_layer = LinearLatentLayer(
            lat_lo, lat_hi, init_latent, name="latent_layer")

        # Single condition [3/5/26]
        self._pipeline   = pipeline
        self._alpha      = alpha
        self._Re         = Re
        self._lat_lo_np  = lat_lo_np
        self._lat_hi_np  = lat_hi_np
        self._penalty_kw = penalty_kwargs
        self._cl_min     = cl_min
        self._fd_eps     = fd_eps
        self._bounds_lam = bounds_lam
        self._initial_lr = None

        self.best_result = None
        self.best_loss   = float("inf")
        self.history_log = []
        self.n_improved  = 0
        self.n_valid     = 0
        self.n_skipped   = 0
        self._n_iters    = 100

    def call(self, inputs=None, training=False):
        z = self.latent_layer(inputs)
        return self.decoder(z, training=False)

    # ------------------------------------------------------------------
    # _compute_loss  [3/5/26: replaces nested loss_at function]
    #
    # Evaluates ONE latent vector at the single (alpha, Re) condition.
    # No loop, no averaging. Just one NeuralFoil call.
    # Returns: (loss_float, info_dict_or_None)
    # ------------------------------------------------------------------
    def _compute_loss(self, z: np.ndarray) -> tuple[float, dict | None]:
        try:
            out = self._pipeline.eval_latent_with_neuralfoil(
                z, alpha=self._alpha, Re=self._Re)
            CL = float(out["CL"])
            CD = float(out["CD"])
            coords = out.get("coords")
        except Exception:
            return float("inf"), None

        if not (np.isfinite(CL) and np.isfinite(CD) and CD > 0):
            return float("inf"), None
        if CL < self._cl_min:
            return float("inf"), None

        # Objective: CD/CL at this single condition
        obj = default_objective(CL, CD)
        if not np.isfinite(obj):
            return float("inf"), None

        # Geometry penalty
        pen = 0.0
        if coords is not None:
            try:
                pen, _ = total_penalty(
                    latent_vec=z, coords=coords, CL=CL,
                    lat_lo=self._lat_lo_np, lat_hi=self._lat_hi_np,
                    **self._penalty_kw)
            except Exception:
                pass

        # Bounds penalty
        bp = self._bounds_lam * float(np.sum(
            np.maximum(0.0, self._lat_lo_np - z) +
            np.maximum(0.0, z - self._lat_hi_np)))

        loss = float(obj + pen + bp)
        info = {"CL": CL, "CD": CD, "obj": obj, "pen": pen, "bp": bp,
                "coords": coords}
        return loss, info

    # ------------------------------------------------------------------
    # train_step  [3/5/26: tf.GradientTape + tf.custom_gradient]
    #
    # Professor's whiteboard:
    #   df/dx = (f(x0) - f(x0+D)) / D      <- finite differences
    #   x_n = x0 - a * df/dx               <- gradient descent update
    #
    # tf.custom_gradient wraps our FD gradients so GradientTape works.
    # ------------------------------------------------------------------
    def train_step(self, data):
        if not hasattr(self, "_t_start"):
            self._t_start = time.time()

        # Cosine LR decay
        it = int(self.optimizer.iterations.numpy())
        n_total = max(getattr(self, "_n_iters", 100), 1)
        if self._initial_lr is None:
            self._initial_lr = float(self.optimizer.learning_rate)
        lr_max = self._initial_lr
        lr_min = lr_max * 0.05
        progress = min(it / n_total, 1.0)
        new_lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))
        self.optimizer.learning_rate.assign(new_lr)

        model = self  # capture for nested function

        # =============================================================
        # tf.custom_gradient: define forward + backward for GradientTape
        # =============================================================
        @tf.custom_gradient
        def forward_pass(w, b):
            # FORWARD: evaluate foil at current w, b
            w_np   = w.numpy().astype(np.float64)
            b_np   = b.numpy().astype(np.float64)
            z_init = model.latent_layer._z_init.numpy().astype(np.float64)
            z_eff  = w_np * z_init + b_np

            loss_val, info = model._compute_loss(z_eff)

            # Store for post-step display
            model._fwd_info = info
            model._fwd_loss = loss_val
            model._fwd_w    = w_np
            model._fwd_b    = b_np
            model._fwd_zi   = z_init

            def grad_fn(dy, variables=None):
                """
                BACKWARD: finite-difference gradients.

                The 'variables' arg is required by tf.custom_gradient because
                the forward pass touches the decoder's frozen weights (they're
                TF Variables even though trainable=False). We return None for
                each variable gradient since we don't want to update them.
                """
                grad_w = np.zeros(6, dtype=np.float64)
                grad_b = np.zeros(6, dtype=np.float64)

                for i in range(6):
                    # [3/5/26] Simplified epsilon
                    eps_w = model._fd_eps / (abs(z_init[i]) + 1e-6)

                    wp = w_np.copy(); wp[i] += eps_w
                    wm = w_np.copy(); wm[i] -= eps_w
                    lp, _ = model._compute_loss(wp * z_init + b_np)
                    lm, _ = model._compute_loss(wm * z_init + b_np)

                    if np.isfinite(lp) and np.isfinite(lm):
                        grad_w[i] = (lp - lm) / (2 * eps_w)
                    elif np.isfinite(lp):
                        grad_w[i] = (lp - loss_val) / eps_w
                    elif np.isfinite(lm):
                        grad_w[i] = (loss_val - lm) / eps_w

                for i in range(6):
                    bp_arr = b_np.copy(); bp_arr[i] += model._fd_eps
                    bm_arr = b_np.copy(); bm_arr[i] -= model._fd_eps
                    lp, _ = model._compute_loss(w_np * z_init + bp_arr)
                    lm, _ = model._compute_loss(w_np * z_init + bm_arr)

                    if np.isfinite(lp) and np.isfinite(lm):
                        grad_b[i] = (lp - lm) / (2 * model._fd_eps)
                    elif np.isfinite(lp):
                        grad_b[i] = (lp - loss_val) / model._fd_eps
                    elif np.isfinite(lm):
                        grad_b[i] = (loss_val - lm) / model._fd_eps

                # Return: (input gradients), [variable gradients]
                # Input gradients = for w and b (our FD results)
                # Variable gradients = None for each frozen decoder weight
                input_grads = (dy * tf.constant(grad_w.astype(np.float32)),
                               dy * tf.constant(grad_b.astype(np.float32)))
                var_grads = [None for _ in (variables or [])]
                return input_grads, var_grads

            return tf.constant(float(loss_val), dtype=tf.float32), grad_fn

        # =============================================================
        # RUN: GradientTape drives the optimization
        # =============================================================
        iter_num = it + 1
        w_saved = self.latent_layer.w.numpy().copy()
        b_saved = self.latent_layer.b.numpy().copy()

        with tf.GradientTape() as tape:
            loss = forward_pass(self.latent_layer.w, self.latent_layer.b)

        if not np.isfinite(self._fwd_loss):
            # FIX: rollback to pre-step w,b so we don't get permanently stuck
            self.latent_layer.w.assign(w_saved)
            self.latent_layer.b.assign(b_saved)
            self.n_skipped += 1
            self._print_step(iter_num, None, {}, skipped=True)
            return {"loss": tf.constant(1000.0, dtype=tf.float32)}

        grads = tape.gradient(loss, [self.latent_layer.w, self.latent_layer.b])
        self.optimizer.apply_gradients(
            zip(grads, [self.latent_layer.w, self.latent_layer.b]))

        # POST-STEP: evaluate new point, record best
        w_new = self.latent_layer.w.numpy().astype(np.float64)
        b_new = self.latent_layer.b.numpy().astype(np.float64)
        z_init = self.latent_layer._z_init.numpy().astype(np.float64)
        z_new = np.clip(w_new * z_init + b_new, self._lat_lo_np, self._lat_hi_np)

        # FIX: sync b to match clipped z_new so next iter's forward pass
        # uses the clipped position, not the raw SGD step that may be out of bounds.
        # b_corrected = z_clipped - w * z_init
        b_corrected = (z_new - w_new * z_init).astype(np.float32)
        self.latent_layer.b.assign(b_corrected)

        new_loss, new_info = self._compute_loss(z_new)
        step_ok = np.isfinite(new_loss) and new_info is not None

        # FIX: Also check if geometry penalty is a hard reject.
        # With SGD, the step can land in a region where the foil shape
        # violates geometry (pen=1000) even though the loss is finite.
        # Without rollback, the optimizer gets permanently stuck there.
        new_pen = 0.0
        if step_ok:
            new_pen = new_info.get("pen", 0.0)
            if new_pen >= 1000.0:
                step_ok = False  # treat as invalid — trigger rollback

        if step_ok:
            self.n_valid += 1
            genuinely_valid = (new_pen < 1.0) and np.isfinite(new_loss)
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

            self.history_log.append({
                "iter":  iter_num,
                "CL":    new_info["CL"],
                "CD":    new_info["CD"],
                "cd_cl": new_info["obj"],
                "loss":  float(new_loss),
            })
        else:
            self.latent_layer.w.assign(w_saved)
            self.latent_layer.b.assign(b_saved)
            self.n_skipped += 1
            self._last_improved = False

        fwd = self._fwd_info or {}
        self._print_step(iter_num, self._fwd_loss, fwd, step_ok=step_ok)
        return {"loss": tf.constant(float(self._fwd_loss), dtype=tf.float32)}

    def _print_step(self, iter_num, loss_0, dbg, skipped=False, step_ok=True):
        n_total = getattr(self, "_n_iters", 100)
        elapsed = time.time() - getattr(self, "_t_start", time.time())
        secs = elapsed / max(iter_num, 1)
        eta = secs * (n_total - iter_num)
        eta_s = f"{eta/3600:.1f}h" if eta >= 3600 else (f"{eta/60:.0f}m" if eta >= 60 else f"{eta:.0f}s")

        if skipped or loss_0 is None:
            print(f"  iter {iter_num:4d}/{n_total}  SKIP  "
                  f"valid={self.n_valid}  skip={self.n_skipped}  ETA {eta_s}")
            return

        CL  = dbg.get("CL", float("nan"))
        CD  = dbg.get("CD", float("nan"))
        ld  = CL / CD if (CD and CD > 0) else 0.0
        pen = dbg.get("pen", 0.0) + dbg.get("bp", 0.0)
        best_ld = 1.0 / max(self.best_loss, 1e-9)
        status = "ok" if step_ok else "SKIP"
        star = " * BEST" if getattr(self, "_last_improved", False) else ""

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
    # Single condition [3/5/26]
    alpha: float = DEFAULT_ALPHA,
    Re: float = DEFAULT_RE,

    n_iters: int = 100,
    tf_learning_rate: float = 0.0005,
    fd_eps: float = 0.01,
    bounds_lam: float = 10.0,

    lam_bounds: float = 1.0,
    lam_geom:   float = 25.0,
    lam_cl:     float = 50.0,

    # Geometry limits - CHANGE THESE for manufacturing (3/5/26)
    min_thickness:     float = 0.006,
    max_thickness:     float = 0.157,
    te_gap_max:        float = 0.01,
    min_max_thickness: float = 0.04,   # <- CHANGE for your 3D printer
    max_camber:        float = 0.08,

    cl_min: float = 0.15,
    cl_max: float | None = None,

    csv_path: str = "data/airfoil_latent_params_6.csv",
    lookup_baseline_path: str = "",
    out_path: str | Path = "outputs",
):
    """Run NOM optimization with tf.GradientTape at a single (alpha, Re)."""

    alpha, Re = snap_condition(alpha, Re)
    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)

    print("=" * 70)
    print("NOM OPTIMIZATION - tf.GradientTape + nom.compile + nom.fit")
    print("=" * 70)
    print(f"Iterations: {n_iters}")
    print(f"Learning rate: {tf_learning_rate}")
    print(f"FD epsilon: {fd_eps}")
    print(f"Condition:  alpha={alpha} deg  Re={Re:,.0f}")
    print(f"Objective:  minimize CD/CL")
    print(f"CL floor:   CL >= {cl_min}")
    print("=" * 70)
    print()

    all_latents = load_latent_dataset(csv_path)
    lat_lo, lat_hi = latent_minmax_bounds(all_latents)
    print(f"Latent bounds from {len(all_latents)} training foils.")

    pipeline = TalarAIPipeline()
    print(f"Pipeline ready (decoder: {pipeline.decoder_path.name})")
    print()

    # Load baseline
    print("=" * 70)
    print("LOADING BASELINE")
    print("=" * 70)
    if not lookup_baseline_path:
        a_s = min(_VALID_ALPHAS, key=lambda a: abs(a - alpha))
        r_s = min(_VALID_RES,    key=lambda r: abs(r - Re))
        tag = f"alpha{a_s:.1f}_Re{r_s:.0e}"
        lookup_baseline_path = f"outputs/best_baseline_foil_{tag}.json"
        print(f"Auto path: {lookup_baseline_path}")

    baseline = load_best_baseline(lookup_baseline_path)
    if baseline is None:
        print("No baseline found. Run build_lookup_table.py first.")
        return

    latent_baseline = np.array(baseline["latent"], dtype=float)
    print(f"Baseline foil:  {baseline.get('filename', '?')}")
    print(f"Baseline latent: {np.round(latent_baseline, 4)}")

    # Evaluate baseline
    print(f"Evaluating baseline at alpha={alpha}, Re={Re:,.0f}...")
    try:
        bl_out = pipeline.eval_latent_with_neuralfoil(
            latent_baseline, alpha=alpha, Re=Re)
        bl_CL = float(bl_out["CL"])
        bl_CD = float(bl_out["CD"])
        bl_coords = bl_out.get("coords")
        bl_cd_cl = default_objective(bl_CL, bl_CD)
        bl_LD = bl_CL / bl_CD if bl_CD > 0 else 0
        print(f"Baseline L/D = {bl_LD:.1f}  (CD/CL = {bl_cd_cl:.6f})")
        print(f"  CL={bl_CL:.4f}  CD={bl_CD:.6f}")
    except Exception as e:
        print(f"Baseline eval failed: {e}")
        bl_cd_cl = float("inf")
        bl_CL = bl_CD = float("nan")
        bl_coords = None
    print()

    # Constraint validation
    penalty_kwargs = dict(
        lam_bounds=lam_bounds, lam_geom=lam_geom, lam_cl=lam_cl,
        min_thickness=min_thickness, max_thickness=max_thickness,
        te_gap_max=te_gap_max, min_max_thickness=min_max_thickness,
        max_camber=max_camber, cl_min=cl_min, cl_max=cl_max,
    )

    if bl_coords is not None and np.isfinite(bl_CL):
        bl_pen, bl_gi = total_penalty(
            latent_vec=latent_baseline, coords=bl_coords, CL=bl_CL,
            lat_lo=lat_lo, lat_hi=lat_hi, **penalty_kwargs)
        if bl_pen >= 1000:
            print("!" * 70)
            print("BASELINE VIOLATES GEOMETRY CONSTRAINTS")
            print(f"  Penalty: {bl_pen}  Reason: {bl_gi.get('reason','?')}")
            print("  FIX: Re-run build_lookup_table.py --phase2")
            print("!" * 70)
            print()
        elif bl_pen > 0:
            print(f"  Baseline soft penalty = {bl_pen:.4f}")
            print()
        else:
            print(f"  Baseline passes all constraints (penalty = 0)")
            print()

    # Build model
    print("=" * 70)
    print("BUILDING NOM MODEL")
    print("=" * 70)
    nom = NOMModel(
        decoder_model=pipeline.decoder,
        lat_lo=lat_lo, lat_hi=lat_hi,
        init_latent=latent_baseline,
        pipeline=pipeline,
        alpha=alpha, Re=Re,
        lat_lo_np=lat_lo, lat_hi_np=lat_hi,
        penalty_kwargs=penalty_kwargs,
        cl_min=cl_min, fd_eps=fd_eps, bounds_lam=bounds_lam,
    )
    nom(None)

    # Init best to baseline
    bl_full_loss = bl_cd_cl
    if bl_coords is not None and np.isfinite(bl_CL):
        try:
            _p, _ = total_penalty(
                latent_vec=latent_baseline, coords=bl_coords, CL=bl_CL,
                lat_lo=lat_lo, lat_hi=lat_hi, **penalty_kwargs)
            _bp = bounds_lam * float(np.sum(
                np.maximum(0, lat_lo - latent_baseline) +
                np.maximum(0, latent_baseline - lat_hi)))
            bl_full_loss = bl_cd_cl + _p + _bp
        except Exception:
            pass
    nom.best_loss = bl_full_loss
    nom.best_result = {
        "latent": latent_baseline.copy(),
        "coords": bl_coords,
        "CL": bl_CL, "CD": bl_CD,
        "cd_cl": bl_cd_cl,
    }

    # nom.summary()
    nom.summary()
    nt = sum(int(np.prod(v.shape)) for v in nom.trainable_variables)
    nf = sum(int(np.prod(v.shape)) for v in nom.non_trainable_variables)
    print(f"\n  Trainable:  {nt} params  (w[6] + b[6])")
    print(f"  Frozen:     {nf} params  (z_init[6] + decoder weights)")
    print(f"  Formula:    z_eff[i] = w[i] * z_init[i] + b[i]")
    print(f"  z_init:     {np.round(latent_baseline, 4)}\n")

    # nom.compile()
    nom.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=tf_learning_rate), #switch .Adam to .SGD to switch 
        run_eagerly=True)
    print(f"nom.compile(Adam(lr={tf_learning_rate}), run_eagerly=True)")
    print()

    # nom.fit()
    print("=" * 70)
    print(f"nom.fit(epochs={n_iters}) - GradientTape + FD")
    print(f"  alpha={alpha} deg  Re={Re:,.0f}")
    print(f"  ~{n_iters * 14:,} NeuralFoil calls total")
    print("=" * 70)
    print()

    dummy = tf.data.Dataset.from_tensors(
        tf.zeros((1, 6), dtype=tf.float32)).repeat()

    nom._n_iters = n_iters
    nom._t_start = time.time()
    nom.fit(dummy, epochs=n_iters, steps_per_epoch=1, verbose=0)

    print(f"\nnom.fit() complete.\n")

    # Save results
    best = nom.best_result
    if best is None or best.get("coords") is None:
        print("NOM found 0 valid candidates.")
        return

    np.savetxt(out_path / "best_latent_nom.csv",
               best["latent"].reshape(1,-1),
               delimiter=",", header="p1,p2,p3,p4,p5,p6", comments="")
    np.save(out_path / "best_latent_nom.npy", best["latent"])
    np.savetxt(out_path / "best_coords_nom.csv",
               best["coords"], delimiter=",", header="x,y", comments="")

    with open(out_path / "nom_history.json", "w") as f:
        json.dump(nom.history_log, f, indent=2)

    best_LD = best["CL"] / best["CD"] if best["CD"] > 0 else 0

    summary = {
        "alpha":             alpha,
        "Re":                Re,
        "n_iters":           int(n_iters),
        "learning_rate":     float(tf_learning_rate),
        "fd_eps":            float(fd_eps),
        "bounds_lam":        float(bounds_lam),
        "lam_bounds":        float(lam_bounds),
        "lam_geom":          float(lam_geom),
        "lam_cl":            float(lam_cl),
        "min_thickness":     float(min_thickness),
        "max_thickness":     float(max_thickness),
        "te_gap_max":        float(te_gap_max),
        "min_max_thickness": float(min_max_thickness),
        "max_camber":        float(max_camber),
        "cl_min":            float(cl_min),
        "cl_max":            None if cl_max is None else float(cl_max),
        "valid_evals":       int(nom.n_valid),
        "skipped":           int(nom.n_skipped),
        "n_improved":        int(nom.n_improved),
        "best_cd_cl":        float(best["cd_cl"]),
        "best_LD":           float(best_LD),
        "best_CL":           float(best["CL"]),
        "best_CD":           float(best["CD"]),
        "best_latent_params": [float(x) for x in best["latent"]],
        "latent_lo":         [float(x) for x in lat_lo],
        "latent_hi":         [float(x) for x in lat_hi],
        "baseline_foil_filename": baseline.get("filename"),
        "final_result_from": "gradienttape_custom_gradient_fd",
    }

    with open(out_path / "nom_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("NOM OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Condition: alpha={alpha} deg  Re={Re:,.0f}")
    print(f"L/D:   {best_LD:.2f}")
    print(f"CD/CL: {best['cd_cl']:.6f}")
    print(f"CL:    {best['CL']:.4f}")
    print(f"CD:    {best['CD']:.6f}")
    print(f"\nValid: {nom.n_valid}  Skipped: {nom.n_skipped}  Improved: {nom.n_improved}")
    print(f"Outputs: {out_path}/")
    print("=" * 70)


if __name__ == "__main__":
    nom_optimize()