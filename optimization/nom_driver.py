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

  This is CLEANER than nom_driver.py because:
    - The finite-difference gradient lives in ONE place (the registered
      custom gradient on the py_function wrapper), not inside train_step
    - train_step itself is a plain GradientTape block with no manual
      gradient computation -- it looks like the TF guide example
    - The forward pass is just: loss = neuralfoil_loss(self.z)
    - tape.gradient(loss, [self.z]) calls our FD grad automatically

  This is what the professor most likely wanted when he said
  "use GradientTape" -- GradientTape managing everything, even if the
  underlying gradient is still finite-difference.

WHY THIS IS BETTER THAN nom_driver.py's @tf.custom_gradient:
  nom_driver.py puts the FD loop INSIDE train_step, which makes
  train_step long and the gradient logic entangled with the step logic.

  This version separates them:
    - NeuralFoilLossOp: a standalone class that wraps NF + FD gradient
    - train_step: just `with tape: loss = nf_op(z)` + tape.gradient
    - Clean, readable, matches the TF guide structure exactly

===========================================================================
STRUCTURE
===========================================================================

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

===========================================================================
MEETING ACTION ITEMS IMPLEMENTED HERE
===========================================================================

[3/12] Delete grad_fn (the standalone one).
    DONE. No standalone grad_fn. The FD gradient lives inside
    NeuralFoilLossOp._wrapped_call as a closure -- that's the right place.

[3/12] Use GradientTape.
    DONE. train_step is a clean GradientTape block, exactly like the
    TF guide "Going lower-level" example.

[3/12] forward_pass cannot return a constant.
    SOLVED by design. NeuralFoilLossOp uses @tf.custom_gradient correctly,
    always returning (tf.constant(value, dtype=tf.float32), grad_fn_closure).

[3/5]  Single (alpha, Re), no averaging.
    DONE.

[2/26] One loop: nom.fit(epochs=n_iters).
    DONE.
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
#
# This class wraps NeuralFoil + constraints into a single callable that:
#   1) Looks like a TF op to GradientTape (via @tf.custom_gradient)
#   2) Computes its own finite-difference gradient internally
#   3) Returns a scalar tf.Tensor that GradientTape can propagate
#
# WHY A CLASS AND NOT A STANDALONE FUNCTION:
#   The operator needs to carry state: pipeline, alpha, Re, bounds, etc.
#   A class with __call__ is the cleanest way to do that while keeping
#   the @tf.custom_gradient pattern self-contained.
# ===========================================================================

class NeuralFoilLossOp:
    """
    Wraps NeuralFoil into a GradientTape-compatible operator.

    Usage:
        nf_op = NeuralFoilLossOp(pipeline, alpha, Re, ...)
        loss_tensor = nf_op(z_tf)   # z_tf is a tf.Variable, shape (6,)
        # GradientTape can then call tape.gradient(loss_tensor, [z_tf])

    The gradient is computed by central finite differences over the 6
    latent dimensions. This is exactly the same math as nom_driver.py,
    but the architecture is cleaner: the FD logic lives HERE, not in
    train_step.
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

        This is the single source of truth for the forward evaluation.
        Both the forward pass and the FD gradient call this function.
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

        Returns a scalar tf.Tensor. When GradientTape calls
        tape.gradient(loss, [z_tf]), TF invokes the grad_fn closure
        below, which returns the central-difference gradient of
        the NeuralFoil + constraint loss w.r.t. z.

        PATTERN: @tf.custom_gradient on a nested function.
        The nested function _wrapped_call is defined here (not at module
        level) so it can capture `self` by closure, giving it access to
        _evaluate(), _fd_eps, etc. without needing global state.

        WHY NESTED AND NOT A METHOD WITH @tf.custom_gradient:
          @tf.custom_gradient needs the decorated function to accept only
          tf.Tensor inputs. If we decorate a method, `self` becomes an
          argument that TF tries to trace -- that breaks things.
          A nested function that closes over `self` avoids this.
        """
        # Capture op reference for closure
        op = self

        @tf.custom_gradient
        def _wrapped_call(z_in):
            # FORWARD
            z_np = z_in.numpy().astype(np.float64)
            loss_val, info = op._evaluate(z_np)
            op.last_info   = info   # store for train_step to read

            # BACKWARD
            def grad_fn(upstream, variables=None):
                """
                Compute d(loss)/dz[i] for i=0..5 via central differences.

                Formula (professor's whiteboard 3/5/26):
                  dL/dz[i] = (L(z + eps*e_i) - L(z - eps*e_i)) / (2*eps)

                Fall-through if one side is invalid:
                  forward difference:  (L(z+eps) - L(z)) / eps
                  backward difference: (L(z) - L(z-eps)) / eps
                  zero if both sides invalid
                """
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
                    # else: both sides invalid -> leave as 0

                input_grad = upstream * tf.constant(grad_z.astype(np.float32))
                # Return None for every variable that is in scope but frozen
                # (e.g. decoder weights). Required by @tf.custom_gradient API.
                var_grads = [None] * len(variables or [])
                return input_grad, var_grads

            # MUST return (tensor, callable) -- this was the 3/12 bug in
            # nom_driver.py where only tf.constant(loss_val) was returned.
            return tf.constant(float(loss_val), dtype=tf.float32), grad_fn

        return _wrapped_call(z_tf)


# ===========================================================================
# NOM MODEL
# ===========================================================================

class NOMModel(tf.keras.Model):
    """
    NOM optimizer using GradientTape + NeuralFoilLossOp.

    KEY DIFFERENCE FROM nom_driver.py:
      train_step is a CLEAN GradientTape block, exactly like the TF guide:

        with tf.GradientTape() as tape:
            loss = self.nf_op(self.z)      # all FD logic hidden inside nf_op
        grads = tape.gradient(loss, [self.z])
        optimizer.apply_gradients(...)

      The finite-difference gradient machinery is encapsulated in
      NeuralFoilLossOp, not scattered through train_step.

    TRAINABLE:  z[6]
    FROZEN:     decoder (held by pipeline, not by this model)
    """

    def __init__(self, nf_op: NeuralFoilLossOp, z_init: np.ndarray,
                 pipeline, alpha, Re, lat_lo_np, lat_hi_np, cl_min):
        super().__init__()

        self.nf_op = nf_op

        # z[6]: directly trainable
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

        # Tracking
        self.best_result   = None
        self.best_loss     = float("inf")
        self.history_log   = []
        self.n_valid       = 0
        self.n_skipped     = 0
        self.n_improved    = 0
        self._n_iters      = 250
        self._t_start      = None
        self._last_improved = False

        # TF guide pattern
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs=None, training=False):
        # Decoder forward pass (not used for gradient here, kept for build())
        z_batched = tf.expand_dims(self.z, axis=0)
        # We don't use the decoder directly -- the pipeline does that
        # internally inside NeuralFoilLossOp._evaluate.
        # Return z for shape compatibility.
        return self.z

    def train_step(self, data):
        """
        Clean GradientTape block, matching the TF guide pattern exactly.

        Compare to the TF guide "Going lower-level" example:

          with tf.GradientTape() as tape:
              y_pred = self(x, training=True)   # forward pass
              loss = keras.losses.mean_squared_error(y, y_pred)
          trainable_vars = self.trainable_variables
          gradients = tape.gradient(loss, trainable_vars)
          self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        Our version:

          with tf.GradientTape() as tape:
              loss = self.nf_op(self.z)         # forward pass (NF + FD grad)
          gradients = tape.gradient(loss, [self.z])
          self.optimizer.apply_gradients(zip(gradients, [self.z]))

        The only difference is that our "forward pass" is a NumPy black box
        with a registered custom gradient. The train_step logic is identical.
        """
        if self._t_start is None:
            self._t_start = time.time()

        it       = int(self.optimizer.iterations.numpy())
        iter_num = it + 1

        # Save z for rollback
        z_saved = self.z.numpy().copy().astype(np.float64)

        # -----------------------------------------------------------------
        # GRADIENT TAPE BLOCK
        # Exactly matches the TF guide "Going lower-level" pattern.
        # The gradient is computed by NeuralFoilLossOp's grad_fn (FD).
        # -----------------------------------------------------------------
        with tf.GradientTape() as tape:
            loss = self.nf_op(self.z)   # calls NF forward + registers FD grad

        fwd_loss = float(loss.numpy())
        fwd_info = self.nf_op.last_info  # set by nf_op during forward pass

        # Skip if forward was invalid (rollback z)
        if not np.isfinite(fwd_loss) or fwd_info is None:
            self.z.assign(z_saved.astype(np.float32))
            self.n_skipped += 1
            self._last_improved = False
            self._print_step(iter_num, fwd_loss, {}, skipped=True)
            self.loss_tracker.update_state(1e9)
            return {m.name: m.result() for m in self.metrics}

        # Compute gradients via tape (invokes FD grad_fn inside nf_op)
        gradients = tape.gradient(loss, [self.z])
        self.optimizer.apply_gradients(zip(gradients, [self.z]))

        # Clip z to latent bounds
        z_new = np.clip(
            self.z.numpy().astype(np.float64),
            self._lat_lo_np,
            self._lat_hi_np,
        )
        self.z.assign(z_new.astype(np.float32))

        # Evaluate new z (post-step) for best tracking
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

            self.history_log.append({
                "iter":  iter_num,
                "CL":    new_info["CL"],
                "CD":    new_info["CD"],
                "cd_cl": new_info["obj"],
                "loss":  float(new_loss),
                "pen":   float(new_info.get("pen", 0.0)),
            })
        else:
            self.z.assign(z_saved.astype(np.float32))
            self.n_skipped += 1
            self._last_improved = False

        self._print_step(iter_num, fwd_loss, fwd_info, step_ok=step_ok)
        self.loss_tracker.update_state(fwd_loss)
        return {m.name: m.result() for m in self.metrics}

    def _print_step(self, iter_num, loss_0, info,
                    skipped=False, step_ok=True):
        n_total  = getattr(self, "_n_iters", 250)
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
    n_iters:          int   = 250,
    tf_learning_rate: float = 0.0005,
    fd_eps:     float = 0.01,
    bounds_lam: float = 10.0,
    lam_bounds: float = 1.0,
    lam_geom:   float = 25.0,
    lam_cl:     float = 50.0,
    min_thickness:     float = 0.006,
    max_thickness:     float = 0.157,
    te_gap_max:        float = 0.005,
    min_max_thickness: float = 0.04,
    max_camber:        float = 0.10,
    cl_min: float = 0.15,
    cl_max: float | None = None,
    csv_path:              str       = "data/airfoil_latent_params_6.csv",
    lookup_baseline_path:  str       = "",
    out_path:              str | Path = "outputs",
):
    """
    Run NOM optimization.

    Same math as nom_driver.py (NeuralFoil + FD gradients), but with
    a cleaner architecture:
      - FD gradient lives in NeuralFoilLossOp, not in train_step
      - train_step is a clean GradientTape block matching the TF guide
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
    print("=" * 70)
    print()

    all_latents = load_latent_dataset(csv_path)
    lat_lo, lat_hi = latent_minmax_bounds(all_latents)

    pipeline = TalarAIPipeline()
    print(f"  Pipeline ready (decoder: {pipeline.decoder_path.name})\n")

    if not lookup_baseline_path:
        a_s = min(_VALID_ALPHAS, key=lambda a: abs(a - alpha))
        r_s = min(_VALID_RES,    key=lambda r: abs(r - Re))
        tag = f"alpha{a_s:.1f}_Re{r_s:.1e}"
        lookup_baseline_path = f"outputs/best_baseline_foil_{tag}.json"

    baseline = load_best_baseline(lookup_baseline_path)
    if baseline is None:
        print("  No baseline. Run build_lookup_table.py first.")
        return

    latent_baseline = np.array(baseline["latent"], dtype=float)
    print(f"  Baseline: {baseline.get('filename', '?')}")
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
        min_thickness=min_thickness, max_thickness=max_thickness,
        te_gap_max=te_gap_max, min_max_thickness=min_max_thickness,
        max_camber=max_camber, cl_min=cl_min, cl_max=cl_max,
    )

    # Build the NeuralFoil loss operator (FD gradient is encapsulated here)
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

    # Initialize best to baseline
    nom.best_loss   = bl_cd_cl
    nom.best_result = {
        "latent": latent_baseline.copy(),
        "coords": bl_coords,
        "CL": bl_CL, "CD": bl_CD, "cd_cl": bl_cd_cl,
    }

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

    # Save results
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

    print("=" * 70)
    print("NOM COMPLETE")
    print("=" * 70)
    print(f"  BASELINE:  L/D={bl_LD:.1f}  CL={bl_CL:.4f}  CD={bl_CD:.6f}")
    print(f"  OPTIMIZED: L/D={best_LD:.1f}  CL={best['CL']:.4f}  CD={best['CD']:.6f}")
    if best_LD > bl_LD and np.isfinite(bl_LD):
        print(f"  IMPROVEMENT: +{(best_LD - bl_LD) / bl_LD * 100:.1f}%")
    print(f"  valid={nom.n_valid}  skipped={nom.n_skipped}  improved={nom.n_improved}")
    print("=" * 70)


if __name__ == "__main__":
    nom_optimize()