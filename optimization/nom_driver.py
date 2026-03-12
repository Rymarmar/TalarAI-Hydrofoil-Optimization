"""
optimization/nom_driver.py

===========================================================================
ARCHITECTURE (professor's whiteboard, 2/26/26)
===========================================================================

  TRAINABLE:
    z[6]  -- the latent vector itself, directly optimized
             starts at z_init (best baseline from lookup table)

  FROZEN:
    Decoder 6->100->1000->80  -> NeuralFoil -> CL, CD

  OBJECTIVE:
    min CD/CL at single (alpha, Re) + penalty

===========================================================================
WHY WE CHANGED FROM  z_eff = w*z_init + b  TO  z  DIRECTLY
===========================================================================

The old formulation z_eff[i] = w[i]*z_init[i] + b[i] had two fatal bugs:

  BUG 1 -- w scales badly when z_init[i] is near zero.
    For e61 baseline: z_init[3] = 0.0145.
    A gradient step of Adam(lr=1e-4) changes w[3] by ~3e-5.
    That moves z_eff[3] by 3e-5 * 0.0145 = 4e-7 -- essentially NOTHING.
    The optimizer could not explore p4 at all.

  BUG 2 -- 12 parameters (w+b) to control 6 degrees of freedom.
    The extra 6 degrees of freedom (w) do not add expressiveness --
    they just interact with z_init and create redundancy that slows
    convergence and makes the gradient landscape harder to navigate.

  FIX: Make z[6] directly trainable. One parameter per latent dimension.
    - Learning rate directly controls how far we step in latent space.
    - No scaling by z_init.
    - All 6 latent dimensions are updated equally and predictably.
    - lr=0.0005 moves z by ~0.0005 per Adam step -- narrow valley needs small steps.

===========================================================================
WHY 500/500 SKIPPED IN THE PREVIOUS RUN
===========================================================================

The e61 baseline has max_camber = 7.08%c, and the constraint limit was
max_camber = 8.0%c -- only 0.92% buffer. Almost every latent perturbation
pushes camber above 8%, triggering a hard reject (penalty=1000).

Combined with the lr=1e-4 that was too small to explore meaningfully,
the optimizer was stuck at z_init for all 500 iterations.

FIX:
  - Direct z parameterization (no w*z_init+b)
  - lr=0.0005 + clipnorm=0.5 (prevents overshooting narrow optimum valley)
  - max_camber raised to 0.10 in default args

===========================================================================
HOW tf.GradientTape WORKS WITH NeuralFoil
===========================================================================

NeuralFoil is a NumPy black box -- TF cannot auto-diff through it.
tf.custom_gradient lets us plug in our own finite-difference gradient
so GradientTape still drives the Adam optimizer:

    @tf.custom_gradient
    def forward_pass(z):
        loss = run_neuralfoil(z.numpy())        # forward (NumPy black box)
        def grad_fn(dy, variables=None):
            grads = finite_differences(z)       # backward (our FD code)
            return dy * grads, [None]*len(vars)
        return loss, grad_fn

    with tf.GradientTape() as tape:
        loss = forward_pass(model.z)
    grads = tape.gradient(loss, [model.z])
    optimizer.apply_gradients(...)

Professor's whiteboard (3/5/26):
    df/dx  = (f(x + D) - f(x - D)) / (2*D)    # central differences
    x_next = x - alpha * df/dx                  # gradient descent step

===========================================================================
MEETING ACTION ITEMS STATUS
===========================================================================

[2/10] Learning rate <= 1e-3.
    DONE. Default tf_learning_rate = 0.0005.
    Note: 0.005 was tried first -- found L/D=143.4 at iter 3 but immediately
    overshot the optimum. Lowered to 0.0005 + clipnorm=0.5 to stay near it.

[2/17] Lookup table as baseline instead of seeds.
    DONE. Baseline loaded from lookup table JSON.

[2/19] nom_optimize for loop: no training. For training: .fit
    DONE. nom.fit() runs the optimization.

[2/26] ONE unified loop -- no separate TF epochs then NOM iterations.
    DONE. nom.fit(epochs=n_iters) = n_iters steps.

[3/5] Take out average conditions -- single (alpha, Re).
    DONE. No loop over conditions.

[3/5] Turn off def loss_at -- restructure.
    DONE. Replaced with _compute_loss() on NOMModel.

[3/5] Simplify epsilon formula.
    DONE. fd_eps is a fixed constant in latent space (not scaled by z_init).

[3/5] USE tf.GradientTape MOST IMPORTANT.
    DONE. train_step uses tf.custom_gradient + tf.GradientTape.
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
DEFAULT_RE    = 150_000

# Valid grid values for snapping (must match build_lookup_table.py)
_VALID_ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
_VALID_RES    = [50_000, 100_000, 150_000, 250_000, 350_000, 450_000]


# ===========================================================================
# HELPERS
# ===========================================================================

def snap_condition(alpha: float, Re: float) -> tuple[float, float]:
    """
    Snap (alpha, Re) to the nearest valid grid value.

    WHY: The lookup table is pre-computed on a discrete grid. If the user
    passes alpha=2.3 or Re=400000, we snap to the nearest grid point so
    we always find a valid baseline foil in the lookup table.
    """
    alpha_s = min(_VALID_ALPHAS, key=lambda a: abs(a - alpha))
    Re_s    = min(_VALID_RES,    key=lambda r: abs(r - Re))
    if alpha_s != alpha or Re_s != Re:
        print(f"  [snap] ({alpha}, Re={Re:.0f}) -> ({alpha_s}, Re={Re_s:.0f})")
    return float(alpha_s), float(Re_s)


def load_latent_dataset(csv_path: str = "data/airfoil_latent_params_6.csv") -> np.ndarray:
    """
    Load 6D latent vectors for all training foils.
    Used to compute lat_lo / lat_hi (the bounds for latent_bounds_penalty).

    RETURNS: (N, 6) float array
    """
    df = pd.read_csv(csv_path)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] != 6:
        raise ValueError(f"Expected 6 latent columns, found {numeric.shape[1]}")
    return numeric.values.astype(float)


def load_best_baseline(json_path: str | Path) -> dict | None:
    """
    Load the best baseline foil entry from the lookup table JSON.

    The JSON must contain at minimum:
      "latent": [p1..p6]
      "filename": str
      "CL": float
      "CD": float
    """
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"  Baseline not found: {json_path}")
        print(f"  Run: python tools/build_lookup_table.py")
        return None
    with open(json_path) as f:
        return json.load(f)


# ===========================================================================
# NOM MODEL  -- z[6] directly trainable (professor's architecture)
# ===========================================================================

class NOMModel(tf.keras.Model):
    """
    NOM optimizer using tf.GradientTape + finite-difference gradients.

    TRAINABLE:   z[6]  -- latent vector, initialized to z_init (baseline)
    FROZEN:      Decoder (6->100->1000->80), z_init is not a tf.Variable

    FORMULA:
      z starts at z_init (best foil from lookup table)
      Each train_step: Adam updates z by -lr * d(loss)/dz
      d(loss)/dz is computed by finite differences (NeuralFoil is NumPy)

    WHY z DIRECTLY (not w*z_init + b):
      - Direct: 6 params control 6 DOF, lr=0.005 means "move 0.005 in
        latent space per Adam step" -- clear and predictable
      - Old w*z_init+b: 12 params for 6 DOF, lr=1e-4 moved z by
        lr * z_init ~= 1e-6 to 2e-5, nearly zero for small z_init values
    """

    def __init__(self, decoder_model, z_init: np.ndarray,
                 pipeline, alpha, Re, lat_lo_np, lat_hi_np,
                 penalty_kwargs, cl_min, fd_eps, bounds_lam):
        super().__init__()

        # Freeze decoder -- we never want gradients flowing into it
        decoder_model.trainable = False
        self.decoder = decoder_model

        # z[6]: the ONLY trainable variable
        # Initialized to z_init (best baseline foil from lookup table)
        z0 = np.array(z_init, dtype=np.float32).reshape(6)
        self.z = self.add_weight(
            name="z",
            shape=(6,),
            initializer=tf.constant_initializer(z0),
            trainable=True,
        )
        # Keep z_init as plain numpy (not a Variable) for reference
        self._z_init_np = z0.astype(np.float64)

        # Operating condition [3/5/26: single condition]
        self._pipeline   = pipeline
        self._alpha      = alpha
        self._Re         = Re
        self._lat_lo_np  = lat_lo_np
        self._lat_hi_np  = lat_hi_np
        self._penalty_kw = penalty_kwargs
        self._cl_min     = cl_min
        self._fd_eps     = fd_eps        # fixed step in latent space
        self._bounds_lam = bounds_lam

        # Tracking
        self.best_result = None
        self.best_loss   = float("inf")
        self.history_log = []
        self.n_improved  = 0
        self.n_valid     = 0
        self.n_skipped   = 0
        self._n_iters    = 250
        self._initial_lr = None

    def call(self, inputs=None, training=False):
        """
        Forward pass through frozen decoder only.
        (NeuralFoil is called separately in _compute_loss, not here.)
        z -> decoder -> y80
        """
        z_batched = tf.expand_dims(self.z, axis=0)
        return self.decoder(z_batched, training=False)

    # ------------------------------------------------------------------
    # _compute_loss  [3/5/26: single condition, no loop]
    #
    # Evaluates ONE latent vector z at the single (alpha, Re) condition.
    # Returns: (loss_float, info_dict_or_None)
    #
    # RETURNS inf (or 1e9) when:
    #   - NeuralFoil crashes
    #   - CL <= cl_min (foil not generating enough lift)
    #   - Geometry hard reject
    #   - Any NaN/Inf in CL or CD
    # ------------------------------------------------------------------
    def _compute_loss(self, z: np.ndarray) -> tuple[float, dict | None]:
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
            # Foil not generating enough lift -- skip
            return 1e9, None

        # Objective: minimize CD/CL at this single condition
        obj = default_objective(CL, CD)
        if not np.isfinite(obj):
            return 1e9, None

        # Geometry penalty (from constraints.py)
        pen = 0.0
        if coords is not None:
            try:
                pen, _ = total_penalty(
                    latent_vec=z, coords=coords, CL=CL,
                    lat_lo=self._lat_lo_np, lat_hi=self._lat_hi_np,
                    **self._penalty_kw)
            except Exception:
                pass

        # Bounds penalty: penalize z going outside training data range
        # This is SEPARATE from the latent_bounds_penalty in total_penalty
        # because we want it applied directly here before the geometry check.
        # lam = bounds_lam (default 10.0) * sum of out-of-bounds violations
        bp = self._bounds_lam * float(np.sum(
            np.maximum(0.0, self._lat_lo_np - z) +
            np.maximum(0.0, z - self._lat_hi_np)))

        loss = float(obj + pen + bp)
        info = {
            "CL": CL, "CD": CD,
            "obj": obj, "pen": pen, "bp": bp,
            "coords": coords,
        }
        return loss, info

    # ------------------------------------------------------------------
    # train_step  [3/5/26: tf.GradientTape + tf.custom_gradient]
    #
    # Professor's whiteboard formula:
    #   df/dz = (f(z + D) - f(z - D)) / (2*D)    <- central differences
    #   z_next = z - lr * df/dz                    <- gradient descent
    #
    # tf.custom_gradient wraps the FD gradient so Adam (via GradientTape)
    # can drive the update automatically.
    #
    # LEARNING RATE SCHEDULE:
    #   - Warmup: flat at lr_max for first 10% of iterations
    #     (lets Adam build up momentum estimates before decaying)
    #   - Cosine decay: lr_max -> lr_max * 0.15 over remaining 90%
    #     (slow decay so late iterations can still explore)
    # ------------------------------------------------------------------
    def train_step(self, data):
        if not hasattr(self, "_t_start"):
            self._t_start = time.time()

        # --- Learning rate schedule (cosine with warmup) ---
        it      = int(self.optimizer.iterations.numpy())
        n_total = max(getattr(self, "_n_iters", 250), 1)
        if self._initial_lr is None:
            self._initial_lr = float(self.optimizer.learning_rate)
        lr_max    = self._initial_lr
        lr_min    = lr_max * 0.15
        warmup_end = int(n_total * 0.10)

        if it < warmup_end:
            new_lr = lr_max
        else:
            progress = min((it - warmup_end) / max(n_total - warmup_end, 1), 1.0)
            new_lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * progress))
        self.optimizer.learning_rate.assign(new_lr)

        # Safe defaults in case forward_pass raises before setting these.
        # Without this, an early exception would cause an AttributeError
        # at line "if not np.isfinite(self._fwd_loss)" below.
        self._fwd_loss = float("inf")
        self._fwd_info = None

        # Save current z before the step so we can rollback if needed
        z_saved = self.z.numpy().copy().astype(np.float64)
        model   = self  # capture for nested function

        # =============================================================
        # tf.custom_gradient: tells GradientTape how to differentiate
        # through our NumPy black-box (NeuralFoil).
        #
        # FORWARD: evaluate loss at current z using NeuralFoil
        # BACKWARD: return finite-difference gradients wrt z
        # =============================================================
        @tf.custom_gradient
        def forward_pass(z_tf):
            # --- FORWARD ---
            z_np = z_tf.numpy().astype(np.float64)
            loss_val, info = model._compute_loss(z_np)

            # Store for post-step display and best-tracking
            model._fwd_info = info
            model._fwd_loss = loss_val
            model._fwd_z    = z_np.copy()

            def grad_fn(dy, variables=None):
                """
                BACKWARD: central finite differences.

                For each latent dimension i:
                    grad[i] = (f(z + eps*e_i) - f(z - eps*e_i)) / (2 * eps)

                where eps = fd_eps (fixed constant in latent space, not
                scaled by z_init -- that was the old bug).

                If only one side is finite (near a boundary), fall back
                to one-sided difference.

                variables arg: required by tf.custom_gradient because the
                frozen decoder weights are TF Variables. We return None
                for each so they never get updated.
                """
                grad_z = np.zeros(6, dtype=np.float64)
                eps    = model._fd_eps

                for i in range(6):
                    # Perturb z in dimension i only
                    zp = z_np.copy(); zp[i] += eps
                    zm = z_np.copy(); zm[i] -= eps

                    lp, _ = model._compute_loss(zp)
                    lm, _ = model._compute_loss(zm)

                    if np.isfinite(lp) and np.isfinite(lm):
                        # Central difference (most accurate)
                        grad_z[i] = (lp - lm) / (2.0 * eps)
                    elif np.isfinite(lp) and np.isfinite(loss_val):
                        # Forward difference (one side failed)
                        grad_z[i] = (lp - loss_val) / eps
                    elif np.isfinite(lm) and np.isfinite(loss_val):
                        # Backward difference (other side failed)
                        grad_z[i] = (loss_val - lm) / eps
                    # else: both sides failed -- leave grad=0 for this dim

                # Scale by upstream gradient dy (standard chain rule)
                # dy is a scalar from GradientTape
                input_grad = dy * tf.constant(grad_z.astype(np.float32))
                var_grads  = [None for _ in (variables or [])]
                return input_grad, var_grads

            return tf.constant(float(loss_val), dtype=tf.float32), grad_fn

        # =============================================================
        # RUN: GradientTape calls forward_pass, then grad_fn via .gradient()
        # =============================================================
        iter_num = it + 1

        with tf.GradientTape() as tape:
            loss = forward_pass(self.z)

        # If forward pass was invalid (NeuralFoil crash, CL too low, etc.)
        # roll back z to where it was before this step.
        if not np.isfinite(self._fwd_loss):
            self.z.assign(z_saved.astype(np.float32))
            self.n_skipped += 1
            self._last_improved = False
            self._print_step(iter_num, None, {}, skipped=True)
            return {"loss": tf.constant(1e9, dtype=tf.float32)}

        # Compute and apply gradients (Adam handles the actual step)
        grads = tape.gradient(loss, [self.z])
        self.optimizer.apply_gradients(zip(grads, [self.z]))

        # --- POST-STEP: clip z to latent bounds ---
        # Clipping keeps z inside the training data range.
        # We do this AFTER apply_gradients so Adam can explore boundaries
        # and the gradient pushes back naturally via bounds_lam penalty,
        # but we hard-clip as a safety net to prevent runaway latent values.
        z_new = np.clip(
            self.z.numpy().astype(np.float64),
            self._lat_lo_np,
            self._lat_hi_np,
        )
        self.z.assign(z_new.astype(np.float32))

        # --- POST-STEP: evaluate new position ---
        new_loss, new_info = self._compute_loss(z_new)
        step_ok = np.isfinite(new_loss) and new_info is not None

        # Hard-reject check: if the post-step geometry penalty is 1000,
        # the foil is physically invalid. Roll back and skip.
        if step_ok:
            new_pen = new_info.get("pen", 0.0)
            if new_pen >= 1000.0:
                step_ok = False

        if step_ok:
            self.n_valid += 1

            # Only count as a genuine improvement if penalty is small
            # (not just a lower loss because we're in a penalized region)
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

            self.history_log.append({
                "iter":  iter_num,
                "CL":    new_info["CL"],
                "CD":    new_info["CD"],
                "cd_cl": new_info["obj"],
                "loss":  float(new_loss),
                "pen":   float(new_info.get("pen", 0.0)),
            })
        else:
            # Roll back z to pre-step value
            self.z.assign(z_saved.astype(np.float32))
            self.n_skipped += 1
            self._last_improved = False

        self._print_step(iter_num, self._fwd_loss,
                         self._fwd_info or {}, step_ok=step_ok)
        return {"loss": tf.constant(float(self._fwd_loss), dtype=tf.float32)}

    def _print_step(self, iter_num, loss_0, dbg,
                    skipped=False, step_ok=True):
        """Print one-line progress for this iteration."""
        n_total = getattr(self, "_n_iters", 250)
        elapsed = time.time() - getattr(self, "_t_start", time.time())
        secs_per = elapsed / max(iter_num, 1)
        eta      = secs_per * (n_total - iter_num)
        eta_s    = (f"{eta/3600:.1f}h" if eta >= 3600 else
                    f"{eta/60:.0f}m"   if eta >= 60   else
                    f"{eta:.0f}s")

        if skipped or loss_0 is None:
            print(f"  iter {iter_num:4d}/{n_total}  SKIP  "
                  f"valid={self.n_valid}  skip={self.n_skipped}  ETA {eta_s}")
            return

        CL   = dbg.get("CL",  float("nan"))
        CD   = dbg.get("CD",  float("nan"))
        ld   = CL / CD if (CD and CD > 0) else 0.0
        pen  = dbg.get("pen", 0.0) + dbg.get("bp", 0.0)
        best_ld = 1.0 / max(self.best_loss, 1e-9)
        status  = "ok" if step_ok else "SKIP"
        star    = " *** BEST" if getattr(self, "_last_improved", False) else ""

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
    # Operating condition [3/5/26: single point, no averaging]
    alpha: float = DEFAULT_ALPHA,
    Re:    float = DEFAULT_RE,

    n_iters:          int   = 250,
    tf_learning_rate: float = 0.0005,  # lr=0.0005 -- lowered from 0.005.
                                        # WHY: at lr=0.005 the optimizer found
                                        # L/D=143.4 at iter 3 but immediately
                                        # overshot it -- Adam's momentum carried
                                        # z far away and it never came back.
                                        # The optimum sits in a narrow valley;
                                        # smaller steps keep z near it.
                                        # (vs old lr=1e-4 with w*z_init+b
                                        #  which moved z by ~1e-6/step -- too small)
    fd_eps:    float = 0.01,           # finite-difference step in latent space
    bounds_lam: float = 10.0,          # weight on out-of-bounds penalty

    # Penalty weights (see constraints.py total_penalty for details)
    # lam_bounds: latent out-of-range
    # lam_geom:   geometry violations (TE gap)
    # lam_cl:     CL out of window
    lam_bounds: float = 1.0,
    lam_geom:   float = 25.0,
    lam_cl:     float = 50.0,

    # Geometry limits  -- CHANGE min_max_thickness for your 3D printer
    min_thickness:     float = 0.006,
    max_thickness:     float = 0.157,
    # [ACTION ITEM 3/5/26] Tightened from 0.01 to 0.005 for manufacturing:
    # A trailing edge gap of 1%c is too open to 3D print cleanly.
    # 0.5%c (0.005) is a tighter but still achievable manufacturing tolerance.
    # The optimizer will now penalize (softly) any TE gap above 0.5%c.
    te_gap_max:        float = 0.005,
    min_max_thickness: float = 0.04,   # <- TUNE: minimum peak thickness
    # max_camber raised to 0.10 from 0.08.
    # WHY: e61 baseline has 7.08% camber. At 0.08 the buffer was only
    # 0.92%, causing EVERY perturbation to be hard-rejected.
    # At 0.10 the buffer is ~3%, giving the optimizer room to explore
    # while still blocking extreme camber (goal: manufacturable foil).
    # Tighten this back toward 0.08 after a good region is found.
    max_camber:        float = 0.10,

    cl_min: float = 0.15,
    cl_max: float | None = None,

    csv_path:              str = "data/airfoil_latent_params_6.csv",
    lookup_baseline_path:  str = "",
    out_path:              str | Path = "outputs",
):
    """
    Run NOM optimization.

    z[6] is optimized directly using Adam + finite-difference gradients.
    Starts from the best foil in the lookup table at (alpha, Re).
    """
    alpha, Re = snap_condition(alpha, Re)
    out_path  = Path(out_path)
    out_path.mkdir(exist_ok=True)

    print("=" * 70)
    print("NOM OPTIMIZATION  --  tf.GradientTape + z directly trainable")
    print("=" * 70)
    print(f"  Iterations:     {n_iters}")
    print(f"  Learning rate:  {tf_learning_rate}  (moves z by ~{tf_learning_rate:.4f}/step)")
    print(f"  FD epsilon:     {fd_eps}  (latent perturbation for gradient)")
    print(f"  Condition:      alpha={alpha} deg  Re={Re:,.0f}")
    print(f"  Objective:      minimize CD/CL  (= maximize L/D)")
    print(f"  CL floor:       CL >= {cl_min}")
    print(f"  Max camber:     {max_camber*100:.1f}%  (raise if all steps rejected)")
    print("=" * 70)
    print()

    # Load training dataset to compute latent bounds
    all_latents = load_latent_dataset(csv_path)
    lat_lo, lat_hi = latent_minmax_bounds(all_latents)
    print(f"  Latent bounds from {len(all_latents)} training foils.")
    for i in range(6):
        print(f"    p{i+1}: [{lat_lo[i]:.4f}, {lat_hi[i]:.4f}]  range={lat_hi[i]-lat_lo[i]:.4f}")
    print()

    # Load pipeline (decoder + NeuralFoil wrapper)
    pipeline = TalarAIPipeline()
    print(f"  Pipeline ready (decoder: {pipeline.decoder_path.name})")
    print()

    # ------------------------------------------------------------------
    # LOAD BASELINE
    # ------------------------------------------------------------------
    print("=" * 70)
    print("LOADING BASELINE")
    print("=" * 70)

    if not lookup_baseline_path:
        # Auto-build path from (alpha, Re) -- matches build_lookup_table.py naming
        a_s = min(_VALID_ALPHAS, key=lambda a: abs(a - alpha))
        r_s = min(_VALID_RES,    key=lambda r: abs(r - Re))
        tag = f"alpha{a_s:.1f}_Re{r_s:.1e}"
        lookup_baseline_path = f"outputs/best_baseline_foil_{tag}.json"
        print(f"  Auto path: {lookup_baseline_path}")

    baseline = load_best_baseline(lookup_baseline_path)
    if baseline is None:
        print("  No baseline found. Run build_lookup_table.py first.")
        return

    latent_baseline = np.array(baseline["latent"], dtype=float)
    print(f"  Baseline foil:   {baseline.get('filename', '?')}")
    print(f"  Baseline latent: {np.round(latent_baseline, 4)}")
    print()

    # Evaluate baseline to confirm it still evaluates correctly
    print(f"  Evaluating baseline at alpha={alpha}, Re={Re:,.0f}...")
    try:
        bl_out  = pipeline.eval_latent_with_neuralfoil(
            latent_baseline, alpha=alpha, Re=Re)
        bl_CL     = float(bl_out["CL"])
        bl_CD     = float(bl_out["CD"])
        bl_coords = bl_out.get("coords")
        bl_cd_cl  = default_objective(bl_CL, bl_CD)
        bl_LD     = bl_CL / bl_CD if bl_CD > 0 else 0.0
        print(f"  Baseline L/D = {bl_LD:.1f}  (CD/CL = {bl_cd_cl:.6f})")
        print(f"    CL={bl_CL:.4f}  CD={bl_CD:.6f}")
    except Exception as e:
        print(f"  Baseline eval failed: {e}")
        bl_cd_cl = 1e9
        bl_CL    = bl_CD = float("nan")
        bl_coords = None
    print()

    # ------------------------------------------------------------------
    # CHECK BASELINE AGAINST CONSTRAINTS
    # Warn loudly if baseline itself is rejected -- this tells you
    # the constraints are too tight before you waste 500 iterations.
    # ------------------------------------------------------------------
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
            # This is a critical warning: if the baseline fails its own
            # constraints, the optimizer starts in a hard-reject zone and
            # every step nearby will also be rejected (500/500 skipped again).
            print("!" * 70)
            print("  WARNING: BASELINE VIOLATES GEOMETRY CONSTRAINTS")
            print(f"  Penalty = {bl_pen}")
            print(f"  Reason  = {bl_gi.get('reason', '?')}")
            if "max_camber_actual" in bl_gi:
                print(f"  Actual camber = {bl_gi['max_camber_actual']*100:.2f}%  "
                      f"limit = {max_camber*100:.2f}%")
            if "t_max" in bl_gi:
                print(f"  Actual t_max  = {bl_gi['t_max']*100:.2f}%  "
                      f"min_max limit = {min_max_thickness*100:.2f}%")
            print()
            print("  FIX options:")
            print("  1. Raise max_camber (e.g. 0.12) if baseline is too cambered")
            print("  2. Lower min_max_thickness if baseline is too thin")
            print("  3. Run build_lookup_table.py --phase2 to pick a different baseline")
            print("!" * 70)
            print()
        elif bl_pen > 0:
            print(f"  Baseline soft penalty = {bl_pen:.4f}  ({bl_gi.get('reason','?')})")
            print()
        else:
            print(f"  Baseline passes all constraints  (penalty = 0)")
            print()

    # ------------------------------------------------------------------
    # BUILD NOM MODEL
    # ------------------------------------------------------------------
    print("=" * 70)
    print("BUILDING NOM MODEL")
    print("=" * 70)

    nom = NOMModel(
        decoder_model=pipeline.decoder,
        z_init=latent_baseline,
        pipeline=pipeline,
        alpha=alpha,
        Re=Re,
        lat_lo_np=lat_lo,
        lat_hi_np=lat_hi,
        penalty_kwargs=penalty_kwargs,
        cl_min=cl_min,
        fd_eps=fd_eps,
        bounds_lam=bounds_lam,
    )
    nom(None)  # build weights

    # Initialize best_result to the baseline so output is never empty
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
    nom.best_loss   = bl_full_loss
    nom.best_result = {
        "latent": latent_baseline.copy(),
        "coords": bl_coords,
        "CL":     bl_CL,
        "CD":     bl_CD,
        "cd_cl":  bl_cd_cl,
    }

    nom.summary()
    n_train = sum(int(np.prod(v.shape)) for v in nom.trainable_variables)
    n_frozen = sum(int(np.prod(v.shape)) for v in nom.non_trainable_variables)
    print(f"\n  Trainable:  {n_train} params  (z[6])")
    print(f"  Frozen:     {n_frozen} params  (decoder weights)")
    print(f"  z starts:   {np.round(latent_baseline, 4)}")
    print(f"  Formula:    z_next = z - lr * d(loss)/dz  (Adam)")
    print()

    # ------------------------------------------------------------------
    # COMPILE AND RUN
    # ------------------------------------------------------------------
    nom.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf_learning_rate,
            clipnorm=0.5,   # Gradient clipping -- caps the gradient magnitude
                            # before Adam applies its step.
                            # WHY: without clipping, a single large gradient
                            # (e.g. when z is near a sharp penalty boundary)
                            # can throw z far away in one step. At lr=0.005
                            # this caused the optimizer to find L/D=143.4 at
                            # iter 3 and then immediately overshoot it, spending
                            # the rest of 500 iters stuck at L/D~79.5.
                            # clipnorm=0.5 means: if the gradient vector's
                            # L2 norm exceeds 0.5, scale it down so the norm
                            # is exactly 0.5. Small gradients are unaffected.
        ),
        run_eagerly=True,   # required: train_step calls .numpy() on TF tensors
    )
    print(f"  nom.compile(Adam(lr={tf_learning_rate}, clipnorm=0.5), run_eagerly=True)")
    print()

    # Each train_step calls NeuralFoil 1 + 2*6 = 13 times:
    #   1 forward + 12 FD probes (2 per latent dimension)
    print("=" * 70)
    print(f"nom.fit(epochs={n_iters})")
    print(f"  ~{n_iters * 13:,} NeuralFoil calls total  "
          f"({13} per step: 1 fwd + 12 FD probes)")
    print("=" * 70)
    print()

    # dummy dataset: train_step ignores the actual data, uses self.z
    dummy = (tf.data.Dataset
             .from_tensors(tf.zeros((1, 6), dtype=tf.float32))
             .repeat())

    nom._n_iters = n_iters
    nom._t_start = time.time()
    nom.fit(dummy, epochs=n_iters, steps_per_epoch=1, verbose=0)

    print(f"\nnom.fit() complete.\n")

    # ------------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------------
    best = nom.best_result
    if best is None or best.get("coords") is None:
        print("NOM found 0 valid candidates. Outputs not saved.")
        return

    # latent vector
    np.savetxt(out_path / "best_latent_nom.csv",
               best["latent"].reshape(1, -1),
               delimiter=",", header="p1,p2,p3,p4,p5,p6", comments="")
    np.save(out_path / "best_latent_nom.npy", best["latent"])

    # coordinates (80x2 for plotting / plotting script)
    np.savetxt(out_path / "best_coords_nom.csv",
               best["coords"], delimiter=",", header="x,y", comments="")

    # history (CL/CD/loss per iteration, for convergence plot)
    with open(out_path / "nom_history.json", "w") as f:
        json.dump(nom.history_log, f, indent=2)

    best_LD = best["CL"] / best["CD"] if best["CD"] > 0 else 0.0

    summary = {
        "alpha":              alpha,
        "Re":                 Re,
        "n_iters":            int(n_iters),
        "learning_rate":      float(tf_learning_rate),
        "fd_eps":             float(fd_eps),
        "bounds_lam":         float(bounds_lam),
        "lam_bounds":         float(lam_bounds),
        "lam_geom":           float(lam_geom),
        "lam_cl":             float(lam_cl),
        "min_thickness":      float(min_thickness),
        "max_thickness":      float(max_thickness),
        "te_gap_max":         float(te_gap_max),
        "min_max_thickness":  float(min_max_thickness),
        "max_camber":         float(max_camber),
        "cl_min":             float(cl_min),
        "cl_max":             None if cl_max is None else float(cl_max),
        "valid_evals":        int(nom.n_valid),
        "skipped":            int(nom.n_skipped),
        "n_improved":         int(nom.n_improved),
        "best_cd_cl":         float(best["cd_cl"]),
        "best_LD":            float(best_LD),
        "best_CL":            float(best["CL"]),
        "best_CD":            float(best["CD"]),
        "best_latent_params": [float(x) for x in best["latent"]],
        "z_init":             [float(x) for x in latent_baseline],
        "latent_lo":          [float(x) for x in lat_lo],
        "latent_hi":          [float(x) for x in lat_hi],
        "baseline_foil_filename": baseline.get("filename"),
        "final_result_from":  "gradienttape_custom_gradient_fd_direct_z",
    }

    with open(out_path / "nom_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------------------------
    print("=" * 70)
    print("NOM OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"  Condition:  alpha={alpha} deg  Re={Re:,.0f}")
    print()
    print(f"  BASELINE (starting point):")
    print(f"    L/D  = {bl_LD:.1f}   (CD/CL = {bl_cd_cl:.6f})")
    print(f"    CL   = {bl_CL:.4f}   CD = {bl_CD:.6f}")
    print()
    print(f"  OPTIMIZED (best found):")
    print(f"    L/D  = {best_LD:.1f}   (CD/CL = {best['cd_cl']:.6f})")
    print(f"    CL   = {best['CL']:.4f}   CD = {best['CD']:.6f}")
    print()
    if best_LD > bl_LD and np.isfinite(bl_LD):
        pct = (best_LD - bl_LD) / bl_LD * 100
        print(f"  IMPROVEMENT: +{pct:.1f}% L/D over baseline")
    elif abs(best_LD - bl_LD) < 0.1:
        print("  No improvement over baseline. See troubleshooting notes below.")
        print()
        print("  TROUBLESHOOTING:")
        print("  1. Check 'skipped' count above. If skipped=n_iters, the")
        print("     constraints are rejecting everything. Try:")
        print(f"     - Raise max_camber (currently {max_camber:.2f}) to 0.12")
        print(f"     - Lower min_max_thickness (currently {min_max_thickness:.3f}) to 0.02")
        print("  2. If valid>0 but no improvement, the baseline may be locally")
        print("     optimal. Try a different starting foil (2nd or 3rd best")
        print("     in the lookup table).")
    print()
    print(f"  Iterations: valid={nom.n_valid}  skipped={nom.n_skipped}  improved={nom.n_improved}")
    print(f"  Outputs:    {out_path}/")
    print("=" * 70)


if __name__ == "__main__":
    nom_optimize()