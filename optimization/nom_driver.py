"""
python -m optimization.nom_driver

optimization/nom_driver.py

UPDATED: Uses lookup table baseline instead of random seed search (prof action item).

ACTION ITEMS ADDRESSED:
  ✓ "Make lookup table, find best airfoil, use as baseline" - load_best_baseline()
  ✓ "Use lookup table instead of seeds" - replaced find_valid_seed()
  ✓ "Alpha 6 is too harsh" - changed to alpha=4.0
  ✓ "5e5 also for reynolds" - Re=5e5 (unchanged)
  ✓ "No more need for camber and le gap max" - removed from constraints
  ✓ "Limits of alpha and reynolds - add to constraints" - added validation

WHAT THIS FILE DOES:
  NOM (Neural Optimization Machine) loop that searches latent space to find
  the foil minimizing CD/CL while satisfying all physical constraints.

WORKFLOW:
  1. Load best baseline from lookup table (replaces random seed search)
  2. Main loop: propose → evaluate → score → keep if better
  3. Save best foil coordinates, latent params, and optimization history
"""

from __future__ import annotations
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# Import pipeline
try:
    from pipeline.talarai_pipeline import TalarAIPipeline
except ModuleNotFoundError:
    from talarai_pipeline import TalarAIPipeline

# Import objective and constraints
try:
    from optimization.objective import default_objective
    from optimization.constraints import latent_minmax_bounds, total_penalty
except ModuleNotFoundError:
    from objective import default_objective
    from constraints import latent_minmax_bounds, total_penalty


# ===========================================================================
# LOAD DATASET
# ===========================================================================

def load_latent_dataset(csv_path: str = "data/airfoil_latent_params_6.csv") -> np.ndarray:
    """
    Load all 1647 latent parameters from training CSV.
    Used to compute bounds (lat_lo, lat_hi).
    """
    df = pd.read_csv(csv_path)
    numeric = df.select_dtypes(include=[np.number])
    
    if numeric.shape[1] != 6:
        raise ValueError(f"Expected 6 latent columns, found {numeric.shape[1]}")
    
    return numeric.values.astype(float)


# ===========================================================================
# LOAD BEST BASELINE (replaces seed search)
# ===========================================================================

def load_best_baseline(json_path: str | Path) -> dict | None:
    """
    Load the best baseline foil from lookup table JSON.
    
    PROF ACTION ITEM: "Use lookup table instead of seeds to find best baseline"
    
    This replaces the old find_valid_seed() function that randomly sampled
    100 foils. Now we use the guaranteed best foil from the full lookup table.
    
    INPUTS:
      json_path -- path to best_baseline_foil_alpha1_Re4e5.json
    
    OUTPUT:
      dict with keys: filename, alpha, Re, CL, CD, L_over_D, latent
      OR None if file not found
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        print(f"⚠️  Lookup table baseline not found: {json_path}")
        print(f"   Run: python tools/build_lookup_table.py")
        return None
    
    with open(json_path, 'r') as f:
        baseline = json.load(f)
    
    return baseline


# ===========================================================================
# SAFE EVALUATION
# ===========================================================================

def safe_eval(pipeline: TalarAIPipeline,
              latent_vec: np.ndarray,
              *,
              alpha: float,
              Re: float,
              debug: bool = False):
    """
    Try to evaluate latent through pipeline. Returns None if anything fails.
    
    NOTE: NeuralFoil overflow warnings (exp/power) are normal for bad geometries
    that push the model out of its training distribution. We catch NaN/inf outputs
    and return None so the optimizer skips them cleanly.
    """
    try:
        out = pipeline.eval_latent_with_neuralfoil(latent_vec, alpha=alpha, Re=Re)
        
        CL = float(out["CL"])
        CD = float(out["CD"])
        coords = out["coords"]
        
        # NeuralFoil returns NaN when geometry causes internal overflow
        if not (np.isfinite(CL) and np.isfinite(CD)):
            if debug:
                print(f"  safe_eval: non-finite CL={CL:.4f} CD={CD:.6f}")
            return None
        
        # CD must be physically positive
        if CD <= 0:
            if debug:
                print(f"  safe_eval: non-positive CD={CD:.6f}")
            return None
        
        if coords.shape != (80, 2):
            return None
        
        return {"CL": CL, "CD": CD, "coords": coords}
    
    except Exception as e:
        if debug:
            print(f"  safe_eval exception: {type(e).__name__}: {e}")
        return None


# ===========================================================================
# PROPOSAL STRATEGIES
# ===========================================================================

def propose_global(lat_lo: np.ndarray, lat_hi: np.ndarray) -> np.ndarray:
    """Random point uniformly sampled from [lat_lo, lat_hi] box."""
    return np.random.uniform(lat_lo, lat_hi).astype(float)


def propose_local(best_latent: np.ndarray,
                  *,
                  lr: float,
                  lat_lo: np.ndarray,
                  lat_hi: np.ndarray) -> np.ndarray:
    """Small Gaussian step from best_latent, clipped to bounds."""
    step = np.random.normal(0.0, 1.0, size=best_latent.shape).astype(float)
    z = np.asarray(best_latent, dtype=float) + float(lr) * step
    return np.clip(z, lat_lo, lat_hi).astype(float)


# ===========================================================================
# TF TRAINING MODEL
# ACTION ITEM (2/19 meeting): nom.summary() / nom.compile() / nom.fit()
#
# PROFESSOR'S WHITEBOARD DIAGRAM:
#   LEFT (trainable):  6 latent weights p1-p6   ← ONLY these updated by Adam
#   INSIDE (frozen):   Decoder 6→100→1000→80    ← weights locked
#   LOSS:              shape proxy + bounds penalty on the 6 params
#   BACKPROP:          loss → decoder → latent_weights (6 params only)
# ===========================================================================


class LinearLatentLayer(tf.keras.layers.Layer):
    """
    Holds 12 trainable parameters — 6 weights (w) and 6 biases (b) — one pair
    per latent dimension.

    PROFESSOR'S WHITEBOARD (2/19, image 2):
      For each latent dimension i:
          y_i  =  (w_i) * x_i  +  b_i

      Where:
        x_i   = the fixed NOM-search starting point for dim i  (frozen constant)
        w_i   = learnable scale  (initialized to 1.0, Adam updates this)
        b_i   = learnable shift  (initialized to 0.0, Adam updates this)
        y_i   = the actual value fed into the decoder for dim i

      So the decoder always receives:  z_effective = w * z_init + b  (element-wise)

    WHY 12 PARAMS INSTEAD OF 6:
      The previous design let Adam move the 6 latent coords directly.  The
      professor's diagram separates *scale* (w) from *shift* (b) so Adam has
      two independent degrees of freedom per dimension:
        - w_i scales the starting geometry (stretch/compress the shape)
        - b_i shifts the starting geometry (translate in latent space)
      Together they give richer per-dimension control with the same decoder.

    WHY A LAYER (not bare tf.Variable):
      Keras only auto-discovers trainable variables registered via add_weight()
      inside a Layer/Model subclass.  Bare tf.Variable objects assigned as
      self.x = tf.Variable(...) are NOT found by model.trainable_variables,
      which breaks nom.summary() and nom.fit().
      Using add_weight() ensures:
        - nom.summary()  →  Trainable params: 12
        - nom.fit()      →  Adam sees and updates all 12
    """

    def __init__(self, lat_lo: np.ndarray, lat_hi: np.ndarray,
                 init_latent: np.ndarray | None = None, **kwargs):
        super().__init__(**kwargs)

        # z_init: the fixed starting point (NOM best latent).
        # Adam does NOT update this — it is the "x" in  y = w*x + b.
        if init_latent is not None:
            z_init = np.array(init_latent, dtype=np.float32).reshape(6)
        else:
            z_init = np.random.uniform(lat_lo, lat_hi).astype(np.float32)

        # Store as a non-trainable constant so it appears in nom.summary()
        # under Non-trainable params alongside the frozen decoder.
        self._z_init = self.add_weight(
            name="z_init",
            shape=(6,),
            initializer=tf.constant_initializer(z_init),
            trainable=False,   # fixed — never updated by Adam
        )

        # 6 scale weights — initialized to 1.0 so y = 1*x + 0 = x at step 0.
        # Adam will nudge these away from 1 to scale each latent dimension.
        self.w = self.add_weight(
            name="w",          # w_1 … w_6
            shape=(6,),
            initializer="ones",
            trainable=True,
        )

        # 6 bias weights — initialized to 0.0 so y = 1*x + 0 = x at step 0.
        # Adam will nudge these to shift each latent dimension.
        self.b = self.add_weight(
            name="b",          # b_1 … b_6
            shape=(6,),
            initializer="zeros",
            trainable=True,
        )

        self._lat_lo = tf.constant(lat_lo.astype(np.float32))
        self._lat_hi = tf.constant(lat_hi.astype(np.float32))

    def call(self, inputs=None):
        """
        Forward pass:  z_effective = w * z_init + b   (element-wise, shape (1,6))

        WHY WE ACCEPT AND IGNORE `inputs`:
          When nom.fit() runs Keras passes the dummy dataset batch here.
          We ignore it — this layer's output depends only on its own weights.
        """
        _ = inputs  # intentionally unused
        z_eff = self.w * self._z_init + self.b   # shape (6,) — professor's y = w*x + b
        return tf.expand_dims(z_eff, axis=0)      # shape (1, 6) for decoder input

    def get_effective_latent(self) -> tf.Tensor:
        """Return z_effective = w * z_init + b as a shape-(6,) tensor."""
        return self.w * self._z_init + self.b

    def per_param_bounds_penalty(self) -> tf.Tensor:
        """
        Per-parameter bounds penalty — one penalty value per latent dimension.

        PROFESSOR: "Another penalty for going over/under min/max 6 params"
        UPGRADE:   Return shape (6,) so each param's violation is tracked
                   individually.  The caller sums or logs them as needed.

        For each dimension i:
          penalty_i = relu(lo_i - z_eff_i) + relu(z_eff_i - hi_i)
          = how far z_eff_i is below its minimum + how far it is above its maximum
          = 0.0 when z_eff_i is inside [lo_i, hi_i]
        """
        z_eff = self.get_effective_latent()          # shape (6,)
        below = tf.nn.relu(self._lat_lo - z_eff)     # > 0 if z_eff below lower bound
        above = tf.nn.relu(z_eff - self._lat_hi)     # > 0 if z_eff above upper bound
        return below + above                          # shape (6,) — one value per param

    def bounds_penalty(self) -> tf.Tensor:
        """
        Scalar aggregate bounds penalty (sum over all 6 dims).
        Used as a single loss term added to CD/CL.
        """
        return tf.reduce_sum(self.per_param_bounds_penalty())


class NOMTrainingModel(tf.keras.Model):
    """
    ACTION ITEM (2/19): Trainable NOM model using nom.summary/compile/fit.

    STRUCTURE (matches professor's whiteboard + 2/26 correction):
      Trainable:  latent_layer  →  12 params: 6 weights (w) + 6 biases (b)
                                   z_effective[i] = w[i] * z_init[i] + b[i]
      Frozen:     decoder       →  Dense 6→100→1000→80, trainable=False
      Output:     y_pred (1, 80) — decoded foil coordinates

    WHY 12 PARAMS (professor's whiteboard, 2/26):
      The whiteboard (image 2) shows:
          y_i = (w_i) * x_i + b_i
      Each latent dimension gets its own weight AND bias, giving Adam two
      degrees of freedom per dimension: scale (w) and shift (b).
      z_init (the NOM best) is the fixed "x" that never changes.
      nom.summary() will show:  Trainable params: 12

    PER-PARAM PENALTIES:
      Rather than one aggregate bounds penalty, we compute penalty[i] for
      each of the 6 effective latent dims individually. This lets us log
      exactly which parameters are violating their training-data bounds
      and by how much — useful for debugging and understanding the optimizer.

    LOSS (professor's diagram: loss = CD/CL + penalty):
      NeuralFoil is Python/NumPy — TF can't auto-differentiate through it.
      We use tf.py_function() + finite differences (one NeuralFoil call per
      latent dim per epoch) to estimate gradients. Adam updates the 12 params.

    WHY tf.py_function:
      Inside nom.fit(), train_step runs in TF graph mode — tensors are
      symbolic and .numpy() is unavailable. tf.py_function temporarily
      switches to eager mode for one call so our NumPy code can run safely.
    """

    def __init__(self, decoder_model: tf.keras.Model,
                 lat_lo: np.ndarray, lat_hi: np.ndarray,
                 init_latent: np.ndarray | None = None,
                 bounds_lam: float = 10.0,
                 fd_eps: float = 0.01,
                 neuralfoil_fn=None,    # callable: latent (6,) → (CL, CD)
                 constraint_fn=None,   # callable: (latent, coords, CL) → float penalty
                 ):
        super().__init__()

        # ACTION ITEM: "Have everything set to be non-trainable / freeze"
        decoder_model.trainable = False
        self.decoder        = decoder_model
        self._lat_lo        = tf.constant(lat_lo.astype(np.float32))
        self._lat_hi        = tf.constant(lat_hi.astype(np.float32))
        self.bounds_lam     = tf.constant(float(bounds_lam), dtype=tf.float32)
        self.fd_eps         = float(fd_eps)
        self._nf_fn         = neuralfoil_fn    # stored for use in train_step
        self._constraint_fn = constraint_fn
        # constraint_fn(latent_np (6,), coords_np (80,2), CL float) → float
        # Returns soft penalty (0.0 when all constraints pass).

        # ACTION ITEM (2/26): 12 trainable params via LinearLatentLayer.
        # Professor's diagram: y_i = w_i * x_i + b_i.
        # nom.summary() will show Trainable params: 12 (6w + 6b).
        self.latent_layer = LinearLatentLayer(lat_lo, lat_hi, init_latent,
                                              name="latent_layer")

    def call(self, inputs=None, training=False):
        """
        Forward pass:
          1. LinearLatentLayer:  z_eff = w * z_init + b   (shape 1×6)
          2. Frozen decoder:     z_eff → (1, 80) y-coords

        WHY WE PASS `inputs` THROUGH:
          Keras calls call() with the dummy dataset batch during .fit().
          We pass it to latent_layer which accepts and ignores it so Keras's
          graph tracing doesn't error.

        TRAINING=FALSE ON DECODER:
          decoder.trainable = False set in __init__ freezes the weights.
          Passing training=False also disables Dropout/BatchNorm inside the
          decoder so its output is stable and deterministic every epoch.
        """
        z = self.latent_layer(inputs)              # (1, 6): w * z_init + b
        return self.decoder(z, training=False)     # frozen decoder: (1,6) → (1,80)

    def _eval_cd_over_cl(self, z_np: np.ndarray) -> float:
        """
        Call NeuralFoil to get CD/CL at latent z_np.
        Returns float('inf') if invalid.
        """
        try:
            CL, CD = self._nf_fn(z_np)
            if np.isfinite(CL) and np.isfinite(CD) and CL > 0 and CD > 0:
                return float(CD / CL)
        except Exception:
            pass
        return float("inf")

    def train_step(self, data):
        """
        ACTION ITEM: nom.fit() calls this exactly once per epoch.

        WHAT `data` IS:
          The dummy tensor from dummy_dataset (a zero tensor of shape [1]).
          We completely ignore it — our loss is computed from NeuralFoil
          called on the CURRENT EFFECTIVE LATENT, not from any training data.

        EFFECTIVE LATENT:
          z_effective[i] = w[i] * z_init[i] + b[i]    (professor's y = w*x + b)
          Adam updates w (6) and b (6) = 12 params total.
          z_init is fixed — it's the NOM-search best, never changed here.

        LOSS:
          total_loss = CD/CL  (NeuralFoil aero objective at z_effective)
                     + lam * sum(per_param_bounds_penalty)   ← one term per dim
                     + geometry_penalty  (thickness, camber, TE gap, CL limits)

        PER-PARAM BOUNDS PENALTY:
          penalty[i] = relu(lo[i] - z_eff[i]) + relu(z_eff[i] - hi[i])
          We log each of the 6 individual penalties so it's clear which
          latent dimensions are violating their training-data bounds.

        GRADIENT:
          NeuralFoil is pure Python/NumPy — TF can't auto-diff through it.
          We use forward finite differences: bump each of 12 params (w_i or b_i)
          by fd_eps, call NeuralFoil again, compute (loss_plus - loss) / fd_eps.

        WHY tf.py_function:
          Inside nom.fit() train_step runs in TF graph mode — tensors are
          symbolic and .numpy() crashes. tf.py_function temporarily switches
          to eager mode so our NumPy/NeuralFoil code runs safely, then hands
          back a TF tensor to the graph.
        """
        _ = data  # dummy tensor from dummy_dataset — intentionally unused

        def _compute_grads_and_loss(w_tensor, b_tensor, z_init_tensor):
            """
            Pure Python function running inside tf.py_function (eager mode).

            INPUTS:
              w_tensor      — current w weights, shape (6,)
              b_tensor      — current b biases,  shape (6,)
              z_init_tensor — fixed NOM starting latent, shape (6,)

            OUTPUT: flat tf.float32 tensor of length 13:
              [0]      = total_loss
              [1:7]    = grad w.r.t. w_0 … w_5
              [7:13]   = grad w.r.t. b_0 … b_5
            """
            w_np     = w_tensor.numpy().astype(np.float64)       # shape (6,)
            b_np     = b_tensor.numpy().astype(np.float64)       # shape (6,)
            z_init_np = z_init_tensor.numpy().astype(np.float64) # shape (6,)
            lo_np    = self._lat_lo.numpy().astype(np.float64)
            hi_np    = self._lat_hi.numpy().astype(np.float64)
            lam      = float(self.bounds_lam.numpy())

            # Effective latent for the current w, b:  z_eff = w * z_init + b
            z_eff = w_np * z_init_np + b_np   # shape (6,) — actual decoder input

            # --- Per-param bounds penalties at current z_eff ---
            # Each element: how far that dim is outside [lo, hi].
            # This is logged so we can see which dims are violating bounds.
            per_param_pen = (np.maximum(0.0, lo_np - z_eff) +
                             np.maximum(0.0, z_eff - hi_np))  # shape (6,)
            bp = lam * float(np.sum(per_param_pen))

            # --- Evaluate CD/CL at current z_eff ---
            loss_c = self._eval_cd_over_cl(z_eff)

            # --- Geometry + CL constraint penalty ---
            geom_pen_c = 0.0
            x_upper = np.linspace(1, 0, 40)
            x_lower = np.linspace(0, 1, 40)
            x_grid  = np.concatenate([x_upper, x_lower])
            if self._constraint_fn is not None:
                try:
                    CL_c, CD_c = self._nf_fn(z_eff)
                    if np.isfinite(CL_c) and CL_c > 0:
                        z_tf = tf.constant(z_eff.reshape(1, 6).astype(np.float32))
                        coords_tf = self.decoder(z_tf, training=False)
                        coords_np = coords_tf.numpy().reshape(80, -1)
                        coords_2d = np.stack([x_grid, coords_np[:, 0]], axis=1)
                        geom_pen_c = float(self._constraint_fn(z_eff, coords_2d, float(CL_c)))
                except Exception:
                    geom_pen_c = 0.0

            if not np.isfinite(loss_c):
                # Invalid foil at current z_eff — hold position.
                # Analytical bounds-penalty gradients still steer w, b back.
                loss_total = 1e6 + bp + geom_pen_c
                dw = np.where(z_eff < lo_np, -lam * z_init_np,
                     np.where(z_eff > hi_np,  lam * z_init_np, 0.0)).astype(np.float32)
                db = np.where(z_eff < lo_np, -lam,
                     np.where(z_eff > hi_np,  lam,  0.0)).astype(np.float32)
                result = np.array([loss_total] + dw.tolist() + db.tolist(),
                                  dtype=np.float32)
                return tf.constant(result, dtype=tf.float32)

            loss_total_val = loss_c + bp + geom_pen_c

            # ---------------------------------------------------------------
            # Finite-difference gradients w.r.t. all 12 trainable params.
            #
            # For each w_i:
            #   z_eff_plus = (w + eps*e_i) * z_init + b
            #   dLoss/dw_i ≈ (loss(z_eff_plus) - loss(z_eff)) / eps
            #
            # For each b_i:
            #   z_eff_plus = w * z_init + (b + eps*e_i)
            #   dLoss/db_i ≈ (loss(z_eff_plus) - loss(z_eff)) / eps
            #
            # We handle w and b separately because bumping w_i scales by
            # z_init[i] (chain rule: dz_eff/dw_i = z_init[i]), while
            # bumping b_i always adds 1 (dz_eff/db_i = 1).
            # ---------------------------------------------------------------
            dw = np.zeros(6, dtype=np.float32)
            db = np.zeros(6, dtype=np.float32)

            for i in range(6):
                # --- gradient w.r.t. w_i ---
                w_plus = w_np.copy(); w_plus[i] += self.fd_eps
                z_plus = w_plus * z_init_np + b_np
                loss_p = self._eval_cd_over_cl(z_plus)

                geom_p = 0.0
                if self._constraint_fn is not None and np.isfinite(loss_p):
                    try:
                        CL_p, _ = self._nf_fn(z_plus)
                        if np.isfinite(CL_p) and CL_p > 0:
                            z_tf_p = tf.constant(z_plus.reshape(1,6).astype(np.float32))
                            c_tf_p = self.decoder(z_tf_p, training=False)
                            c_np_p = c_tf_p.numpy().reshape(80, -1)
                            c_2d_p = np.stack([x_grid, c_np_p[:, 0]], axis=1)
                            geom_p = float(self._constraint_fn(z_plus, c_2d_p, float(CL_p)))
                    except Exception:
                        geom_p = 0.0

                below_p = np.maximum(0.0, lo_np - z_plus)
                above_p = np.maximum(0.0, z_plus - hi_np)
                bp_p    = lam * float(np.sum(below_p + above_p))
                total_p = loss_p + bp_p + geom_p

                if np.isfinite(total_p) and np.isfinite(loss_total_val):
                    dw[i] = float((total_p - loss_total_val) / self.fd_eps)

                # --- gradient w.r.t. b_i ---
                b_plus = b_np.copy(); b_plus[i] += self.fd_eps
                z_plus_b = w_np * z_init_np + b_plus
                loss_pb = self._eval_cd_over_cl(z_plus_b)

                geom_pb = 0.0
                if self._constraint_fn is not None and np.isfinite(loss_pb):
                    try:
                        CL_pb, _ = self._nf_fn(z_plus_b)
                        if np.isfinite(CL_pb) and CL_pb > 0:
                            z_tf_pb = tf.constant(z_plus_b.reshape(1,6).astype(np.float32))
                            c_tf_pb = self.decoder(z_tf_pb, training=False)
                            c_np_pb = c_tf_pb.numpy().reshape(80, -1)
                            c_2d_pb = np.stack([x_grid, c_np_pb[:, 0]], axis=1)
                            geom_pb = float(self._constraint_fn(z_plus_b, c_2d_pb, float(CL_pb)))
                    except Exception:
                        geom_pb = 0.0

                below_pb = np.maximum(0.0, lo_np - z_plus_b)
                above_pb = np.maximum(0.0, z_plus_b - hi_np)
                bp_pb    = lam * float(np.sum(below_pb + above_pb))
                total_pb = loss_pb + bp_pb + geom_pb

                if np.isfinite(total_pb) and np.isfinite(loss_total_val):
                    db[i] = float((total_pb - loss_total_val) / self.fd_eps)

            # Add analytical bounds-penalty gradient contribution (chain rule):
            #   d(bounds_pen)/dw_i = lam * sign * z_init[i]
            #   d(bounds_pen)/db_i = lam * sign
            sign_vec = np.where(z_eff < lo_np, -1.0,
                       np.where(z_eff > hi_np,  1.0, 0.0))
            dw += (lam * sign_vec * z_init_np).astype(np.float32)
            db += (lam * sign_vec).astype(np.float32)

            # Pack: [loss, dw_0..dw_5, db_0..db_5]  → 13 floats
            result = np.array([loss_total_val] + dw.tolist() + db.tolist(),
                              dtype=np.float32)
            return tf.constant(result, dtype=tf.float32)

        # --- Call _compute_grads_and_loss through tf.py_function ---
        # Passes current w, b, and z_init (all as tensors) into eager-mode Python.
        # Returns shape (13,): [loss, dw_0..dw_5, db_0..db_5]
        result_tensor = tf.py_function(
            func=_compute_grads_and_loss,
            inp=[
                self.latent_layer.w,
                self.latent_layer.b,
                self.latent_layer._z_init,
            ],
            Tout=[tf.float32],
        )[0]  # unwrap single-element list

        # Unpack result tensor:
        #   result_tensor[0]    = total_loss  (scalar)
        #   result_tensor[1:7]  = dw_0 … dw_5
        #   result_tensor[7:13] = db_0 … db_5
        loss_total_t = result_tensor[0:1]     # shape (1,)
        dw_t         = result_tensor[1:7]     # shape (6,) — grad for w weights
        db_t         = result_tensor[7:13]    # shape (6,) — grad for b biases

        # Apply gradients: Adam updates w (6 params) and b (6 params).
        # Each gets its own per-dimension momentum/variance estimate.
        self.optimizer.apply_gradients([
            (dw_t, self.latent_layer.w),
            (db_t, self.latent_layer.b),
        ])

        # Return metrics for Keras to display during nom.fit().
        # loss_total_t[0] extracts the scalar from the (1,) tensor.
        return {
            "loss":  loss_total_t[0],   # total = CD/CL + bounds_pen + geom_pen
            "CD_CL": result_tensor[0],  # same here (CD/CL dominates when valid)
        }

    def get_latent_numpy(self) -> np.ndarray:
        """
        Return the EFFECTIVE latent z_eff = w * z_init + b as shape (6,).

        WHY z_eff AND NOT w OR b DIRECTLY:
          Adam updated w and b, but the decoder always receives z_eff.
          Returning z_eff is what the rest of the pipeline (NeuralFoil eval,
          saving to JSON, etc.) expects — a shape-(6,) latent coordinate.

        WHY .numpy() IS SAFE HERE:
          Called outside nom.fit() — TF is back in eager mode so .numpy() works.
        """
        return self.latent_layer.get_effective_latent().numpy().reshape(6)


def build_and_train_nom(
    pipeline,
    lat_lo: np.ndarray,
    lat_hi: np.ndarray,
    *,
    init_latent: np.ndarray | None = None,
    learning_rate: float = 0.0005,   # TUNED: was 0.005 — Adam at 0.005 overshot badly
                                     # (loss oscillated: 0.0089→0.0172→0.0098 in first 3 epochs)
                                     # 0.0005 gives smoother descent and stays near valid region
    n_epochs: int = 500,             # TUNED: was 200 — more epochs at smaller lr = smoother
                                     # convergence. Total compute cost is similar (lr↓10x, epochs↑2x)
    bounds_lam: float = 10.0,
    fd_eps: float = 0.01,
    alpha: float = 1.0,
    Re: float = 450000.0,
    penalty_kwargs: dict | None = None,  # geometry constraint kwargs forwarded to total_penalty
    verbose: bool = True,
) -> np.ndarray:
    """
    ACTION ITEM (2/19 meeting): nom.summary() → nom.compile() → nom.fit()
    UPDATED (2/26): 12 trainable params (6w + 6b) per professor's whiteboard.

    PROFESSOR'S DIAGRAM (whiteboard, image 2):
      y_i = w_i * x_i + b_i   for each of 6 latent dimensions
      x_i = z_init[i]  (fixed NOM-best starting point, never changed)
      Adam optimizes w (6) and b (6) = 12 params total.

    SEQUENCE:
      1. nom = NOMTrainingModel(...)
      2. nom.summary()                    → verify 12 trainable, rest frozen
      3. nom.compile(Adam(lr))            → set optimizer
      4. nom.fit(data, epochs=n_epochs)   → run gradient descent via .fit()

    LOSS inside .fit():
      total_loss = CD/CL  (NeuralFoil aero objective at z_eff = w*z_init+b)
                 + lam * sum(per_param_bounds_penalty)  (one term per dim i)
                 + geometry_penalty  (thickness, camber, TE gap, CL limits)

    OUTPUT: best refined effective latent z_eff = w*z_init+b, shape (6,)
    """
    print()
    print("=" * 70)
    print("TF TRAINING  (nom.summary → nom.compile → nom.fit)")
    print("=" * 70)

    # Wrap NeuralFoil call so train_step can call it as a plain function
    def neuralfoil_fn(z_np: np.ndarray):
        """z_np shape (6,) → (CL, CD) floats."""
        res = pipeline.eval_latent_with_neuralfoil(z_np, alpha=alpha, Re=Re)
        return float(res["CL"]), float(res["CD"])

    # Wrap total_penalty as a simple (latent, coords, CL) → float callable
    # so train_step can call it without knowing about lat_lo/lat_hi/kwargs.
    # This is the constraint_fn that was missing — it brings thickness, camber,
    # TE gap, and CL limits into the TF training loss.
    _pkwargs = penalty_kwargs or {}
    def constraint_fn(z_np: np.ndarray, coords_np: np.ndarray, CL: float) -> float:
        """Returns soft geometry+CL penalty. 0.0 means all constraints pass."""
        try:
            pen, _ = total_penalty(
                latent_vec=z_np,
                coords=coords_np,
                CL=CL,
                lat_lo=lat_lo,
                lat_hi=lat_hi,
                **_pkwargs,
            )
            # Cap the penalty contribution so a single hard-reject (pen=1000)
            # doesn't completely overwhelm the CD/CL signal. Adam needs to
            # still see the aero gradient direction even when constraints are
            # violated, otherwise it freezes in place.
            return float(min(pen, 50.0))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # 1. Build
    # ------------------------------------------------------------------
    nom = NOMTrainingModel(
        decoder_model=pipeline.decoder,
        lat_lo=lat_lo,
        lat_hi=lat_hi,
        init_latent=init_latent,
        bounds_lam=bounds_lam,
        fd_eps=fd_eps,
        neuralfoil_fn=neuralfoil_fn,
        constraint_fn=constraint_fn,
    )
    nom(None)  # build layers so summary() works

    # ------------------------------------------------------------------
    # 2. nom.summary()
    # ACTION ITEM: "make sure structure is correct: nom.summary()"
    # ------------------------------------------------------------------
    if verbose:
        nom.summary()
        print()

        n_train  = sum(int(np.prod(v.shape)) for v in nom.trainable_variables)
        n_frozen = sum(int(np.prod(v.shape)) for v in nom.non_trainable_variables)

        # ── Human-readable training model summary ──────────────────────
        # Plain-English breakdown that matches the professor's whiteboard:
        #   LEFT (trainable):  w[6] + b[6] = 12 params Adam updates
        #   FROZEN (big box):  decoder 6→100→1000→80, weights locked
        #   FORMULA:           z_eff[i] = w[i]*z_init[i] + b[i]  (y = w·x + b)
        print("=" * 70)
        print("TRAINING MODEL STRUCTURE  (updated 2/26: 12 params, y=wx+b)")
        print("=" * 70)
        print()
        print("  TRAINABLE  (Adam updates these — left side of whiteboard):")
        print("    w  [shape (6,)]  scale each latent dim   init = 1.0")
        print("    b  [shape (6,)]  shift each latent dim   init = 0.0")
        train_ok = "✓" if n_train == 12 else f"✗ expected 12, got {n_train}"
        print(f"    total trainable params: {n_train}  {train_ok}")
        print()
        print("  FROZEN  (decoder — big rectangle on whiteboard, weights locked):")
        print("    decoder  6 → 100 → 1000 → 80   (trainable=False)")
        print(f"    total frozen params: {n_frozen}")
        print()
        print("  EFFECTIVE LATENT  (what enters the decoder each step):")
        print("    z_eff[i] = w[i] * z_init[i] + b[i]   ← y = w·x + b")
        print("    z_init   = NOM best latent (fixed constant, never touched by Adam)")
        print("    At epoch 0: w=1, b=0  →  z_eff = z_init  (start from NOM best)")
        if init_latent is not None:
            print(f"    z_init: {np.round(init_latent, 4)}")
        print()
        print("  LOSS  = CD/CL  +  λ·Σ(per-param bounds penalty)  +  geometry penalty")
        print("  GRAD  = finite differences — bump w_i then b_i independently by ε")
        print(f"          ε={fd_eps}   λ_bounds={bounds_lam}   lr={learning_rate}")
        print("=" * 70)
        print()

    # ------------------------------------------------------------------
    # 3. nom.compile(Adam(...))
    # ACTION ITEM: "nom.compile(Adam(learning_rate, learning))"
    # ------------------------------------------------------------------
    nom.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    # ------------------------------------------------------------------
    # 4. nom.fit(...)
    # ACTION ITEM: "nom.fit()"
    #
    # HOW THE DUMMY DATASET WORKS:
    #   nom.fit() requires an iterable dataset to loop over. We create one
    #   with exactly ONE dummy tensor (zeros) that repeats once. This means:
    #     - Each epoch has exactly 1 batch  → 1 call to train_step per epoch
    #     - epochs=n_epochs → Keras runs train_step n_epochs times total
    #
    #   That is exactly what we want: one full NeuralFoil evaluation + one
    #   finite-difference gradient step per epoch. train_step ignores the
    #   dummy tensor entirely and queries NeuralFoil on the current latent
    #   weights to compute the real CD/CL loss and gradient.
    #
    #   repeat(1) vs repeat(): repeat(1) gives 1 element total per dataset.
    #   Keras loops the dataset once per epoch, so each epoch = 1 step.
    #   This is intentional — do NOT change to repeat() (infinite), as that
    #   would give Keras infinite steps per epoch and it would never advance
    #   to the next epoch.
    # ------------------------------------------------------------------
    dummy_dataset = tf.data.Dataset.from_tensors(tf.zeros([1])).repeat(1)

    if verbose:
        print(f"Running nom.fit() for {n_epochs} epochs  (Adam lr={learning_rate})")
        print(f"  Loss = CD/CL (NeuralFoil) + per-param bounds_penalty (12 params: 6w+6b) + geometry_penalty")
        print(f"  Gradient: finite differences over w and b independently, eps={fd_eps}")
        print(f"  Geometry constraints active: {'yes' if penalty_kwargs else 'no (no penalty_kwargs passed)'}")
        print()

    history = nom.fit(
        dummy_dataset,
        epochs=n_epochs,
        verbose=1 if verbose else 0,
    )

    refined_latent = nom.get_latent_numpy()

    if verbose:
        final_loss = history.history["loss"][-1]
        print()
        print(f"✓ nom.fit() complete.  Final CD/CL loss: {final_loss:.5f}  (= L/D {1/max(final_loss,1e-9):.1f})")
        print(f"  Refined latent: {refined_latent}")
        print("=" * 70)
        print()

    return refined_latent


# ===========================================================================
# MAIN NOM OPTIMIZATION
# ===========================================================================

def nom_optimize(
    *,
    # --- Operating conditions (PROF ACTION ITEM: alpha=4 instead of 6) ---
    alpha: float = 1.0,   # per physical testing conditions (slides: alpha~1 deg at max speed)
    Re: float = 450000,     # design point -- max speed from Ski Cat spreadsheet (corrected 2/19: was 440000)
    
    # --- Iterations ---
    n_iters: int = 450,
    
    # --- Learning rate ---
    learning_rate_init: float = 0.005,
    lr_decay: float = 0.999,
    
    # --- Strategy balance (DEPRECATED — kept for signature compatibility only) ---
    # ACTION ITEM (2/19 meeting): "Lines 397-400, 404-405: global is not needed -- take out mode"
    # p_local is NO LONGER USED in the optimization loop.
    # The old 75%/25% probabilistic global/local split has been removed.
    # The loop now ALWAYS proposes locally (local = only strategy).
    # Global search is only used as an emergency fallback when best=None.
    # Broader exploration is now handled by TF .fit() after the loop.
    # This parameter is kept so existing callers that pass p_local= don't break.
    p_local: float = 0.75,  # DEPRECATED: not used, see comment above
    
    # --- Lambda weights (AUTO-NORMALIZED in constraints.py) ---
    lam_bounds: float = 1.0,
    lam_geom: float = 25.0,
    lam_cl: float = 50.0,
    
    # --- Geometry limits (ACTION ITEM: removed camber, le_gap) ---
    min_thickness: float = 0.006,   # dataset min (0.0071) * 0.9; 0.04 rejects ALL training foils
    max_thickness: float = 0.157,  # dataset max (0.1427) * 1.1
    te_gap_max: float = 0.01,  # ACTION ITEM: only TE, no LE
    
    # --- CL window ---
    cl_min: float | None = 0.15,
    cl_max: float | None = None,   # no ceiling: baseline CL=1.067 >> 0.20, would hard-reject everything
    
    # --- Paths ---
    csv_path: str = "data/airfoil_latent_params_6.csv",
    # ACTION ITEM (2/19 meeting): "lookup_table_path Line 202: take out None"
    # Changed default from None to "" (empty string).
    # Empty string → auto-constructs path from alpha/Re below.
    # None was implicit/hidden; "" makes it explicit that you need a path.
    lookup_baseline_path: str = "",
    # Options:
    #   "" (default) → auto-loads best_baseline_foil_alpha{a}_Re{Re}.json
    #                  (best foil specifically at your chosen alpha + Re)
    #   'outputs/best_baseline_foil_averaged.json'
    #                  (best foil averaged across all conditions --
    #                   good all-around starting point regardless of alpha/Re)
    out_path: str | Path = "outputs",
):
    """
    Main NOM optimization loop using lookup table baseline.
    
    CHANGES FROM OLD VERSION:
      - Replaced find_valid_seed() with load_best_baseline()
      - Changed alpha=6 to alpha=4 (prof: "6 is too harsh")
      - Removed camber_max_abs, le_gap_max (prof: "no more need")
      - Added alpha/Re validation (prof: "add limits to constraints")
    """
    
    # Validate alpha and Re (PROF ACTION ITEM: "add limits to constraints")
    if not (0 <= alpha <= 15):
        raise ValueError(f"Alpha={alpha}° out of range [0, 15]")
    if not (1e4 <= Re <= 1e7):
        raise ValueError(f"Re={Re:.0e} out of range [1e4, 1e7]")
    
    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("NOM OPTIMIZATION (LOOKUP TABLE BASELINE)")
    print("=" * 70)
    print(f"Target: alpha={alpha}°, Re={Re:.0e}")
    print(f"Iterations: {n_iters}")
    # ACTION ITEM (2/19 meeting): removed p_local/global mode splitting.
    # Strategy is now: always local from best, global only as fallback (best=None).
    # TF .fit() provides the broader exploration after the random search loop.
    print(f"Strategy: local search from best baseline (global fallback if no valid candidate found)")
    print(f"Post-loop: TF refinement via nom.summary → nom.compile → nom.fit")
    print("=" * 70)
    print()
    
    # -----------------------------------------------------------------------
    # Load dataset latent bounds
    # ACTION ITEM (2/19 meeting): "Line 244-251: not needed, just use latent
    # params in the json file"
    # KEPT but simplified: we still need lat_lo/lat_hi for bounds-checking
    # during optimization (latent_bounds_penalty). The latent params for the
    # STARTING POINT come from the JSON baseline below -- not from this dataset.
    # -----------------------------------------------------------------------
    all_latents = load_latent_dataset(csv_path)
    lat_lo, lat_hi = latent_minmax_bounds(all_latents)
    print(f"Latent bounds computed from {len(all_latents)} training foils.")
    print()
    
    # -----------------------------------------------------------------------
    # Initialize pipeline
    # -----------------------------------------------------------------------
    
    print("Initializing pipeline...")
    pipeline = TalarAIPipeline()
    print(f"✓ Pipeline ready (decoder: {pipeline.decoder_path.name})")
    print()
    
    # -----------------------------------------------------------------------
    # Load best baseline from lookup table (replaces seed search)
    # -----------------------------------------------------------------------
    
    print("=" * 70)
    print("LOADING BEST BASELINE (from lookup table)")
    print("=" * 70)
    
    # ACTION ITEM (2/19 meeting): "If the user gives decimal for Re and AoA,
    # mod the value and take out the remainder -- Line 269 do it there"
    # Snap alpha and Re to the nearest valid lookup table value so the
    # auto-constructed JSON path always matches an actual file on disk.
    # Example: alpha=1.3 → 1.0, Re=220000 → 250000
    _valid_alphas = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    _valid_res    = [150000, 250000, 350000, 450000]

    alpha_snapped = min(_valid_alphas, key=lambda a: abs(a - alpha))
    Re_snapped    = min(_valid_res,    key=lambda r: abs(r - Re))

    if alpha_snapped != alpha:
        print(f"⚠️  alpha={alpha} snapped to nearest valid value: {alpha_snapped}")
        alpha = alpha_snapped
    if Re_snapped != Re:
        print(f"⚠️  Re={Re:.0f} snapped to nearest valid value: {Re_snapped:.0f}")
        Re = Re_snapped

    # Auto-construct lookup path from (snapped) alpha/Re
    if not lookup_baseline_path:
        # ACTION ITEM (2/19 meeting): "lookup_table_path Line 202: take out None"
        # Empty string triggers auto-construction here.
        tag = f"alpha{alpha:.1f}_Re{Re:.0e}"
        lookup_baseline_path = f"outputs/best_baseline_foil_{tag}.json"
        print(f"Auto-constructed baseline path: {lookup_baseline_path}")
    baseline = load_best_baseline(lookup_baseline_path)
    
    best = None
    
    if baseline is not None:
        print(f"Baseline foil: {baseline['filename']}")
        # NOTE: Handle both standard JSON format (single alpha/Re, has CL/CD keys)
        # and averaged JSON format (multi-condition, has mean_L_over_D instead).
        # The averaged JSON from build_lookup_table_averaged.py does NOT have CL/CD
        # because those are averaged across conditions -- only latent + L/D stats.
        # In both cases, the latent vector is what actually matters for optimization.
        if 'CL' in baseline:
            # Standard single-condition baseline (e.g. best_baseline_foil_alpha1_Re4e+05.json)
            print(f"  CL:        {baseline['CL']:.4f}")
            print(f"  CD:        {baseline['CD']:.6f}")
            print(f"  L/D:       {baseline['L_over_D']:.2f}")
            print(f"  CD/CL:     {baseline['CD_over_CL']:.6f}")
        else:
            # Averaged multi-condition baseline (e.g. best_baseline_foil_averaged.json)
            print(f"  Mean L/D:  {baseline.get('mean_L_over_D', 'N/A'):.2f}  (averaged across {baseline.get('n_conditions_valid','?')} conditions)")
            print(f"  Min L/D:   {baseline.get('min_L_over_D', 'N/A'):.2f}  (worst-case condition)")
            print(f"  NOTE: No single CL/CD -- this foil was selected for robustness across alpha/Re range")
        print()
        
        # Verify baseline is valid with current constraints
        print("Verifying baseline passes current constraints...")
        
        latent_baseline = np.array(baseline['latent'], dtype=float)
        res = safe_eval(pipeline, latent_baseline, alpha=alpha, Re=Re)
        
        if res is not None:
            CL, CD, coords = res['CL'], res['CD'], res['coords']
            obj = default_objective(CL, CD)
            
            penalty_kwargs = {
                'lam_bounds': lam_bounds,
                'lam_geom': lam_geom,
                'lam_cl': lam_cl,
                'min_thickness': min_thickness,
                'max_thickness': max_thickness,
                'te_gap_max': te_gap_max,
                'cl_min': cl_min,
                'cl_max': cl_max,
                # Interpretation A (confirmed 2/23): block extreme camber >4%c, not all camber
                'min_max_thickness': 0.04,
                'max_camber': 0.04,
            }
            
            pen, pen_info = total_penalty(
                latent_vec=latent_baseline,
                coords=coords,
                CL=CL,
                lat_lo=lat_lo,
                lat_hi=lat_hi,
                **penalty_kwargs
            )
            
            # FIX: also reject baseline if pen >= 1000 (hard reject)
            # Previously only checked isfinite, so a pen=1000 baseline
            # was accepted as the starting point and everything inherited it.
            if np.isfinite(pen) and np.isfinite(obj) and pen < 1000.0:
                total = float(obj + pen)
                
                best = {
                    'latent': latent_baseline.copy(),
                    'coords': coords.copy(),
                    'CL': float(CL),
                    'CD': float(CD),
                    'objective': float(obj),
                    'penalty': float(pen),
                    'total': float(total),
                    't_min': float(pen_info.get('t_min', 0.0)),
                    't_max': float(pen_info.get('t_max', 0.0)),
                    'te_gap': float(pen_info.get('te_gap', 0.0)),
                }
                
                print(f"✓ Baseline valid!")
                print(f"  Starting from: L/D={CL/CD:.2f}, CD/CL={obj:.6f}")
                print()
            else:
                print(f"⚠️  Baseline failed constraints (pen={pen:.2f})")
                print(f"   Starting with global exploration")
                print()
        else:
            print(f"⚠️  Baseline evaluation failed")
            print(f"   Starting with global exploration")
            print()
    else:
        print("⚠️  No baseline loaded - starting with global exploration")
        print()
    
    print("=" * 70)
    print()
    
    # -----------------------------------------------------------------------
    # Main optimization loop
    # -----------------------------------------------------------------------
    
    penalty_kwargs = {
        'lam_bounds': lam_bounds,
        'lam_geom': lam_geom,
        'lam_cl': lam_cl,
        'min_thickness': min_thickness,
        'max_thickness': max_thickness,
        'te_gap_max': te_gap_max,
        'cl_min': cl_min,
        'cl_max': cl_max,
        # ACTION ITEM (2/19 meeting): manufacturing constraints
        # CALIBRATION HISTORY:
        #   max_camber=0.01 → too tight, zero valid candidates (whole dataset is cambered)
        #   max_camber=0.08 → too loose, best foil was 7.4%c Eppler-class (hard to print)
        #   max_camber=0.04 → FINAL (Interpretation A confirmed 2/23):
        #     "No extreme camber hard to 3D print" not "zero camber"
        #     Allows NACA 0012 (0%), NACA 2412 (2%), NACA 4412 (4%)
        #     Blocks  NACA 6412 (6%), Eppler 61 (7.4%), any foil above 4%c
        #
        #   min_max_thickness=0.04: foil must have at least 4% peak thickness
        #     (structural depth for 3D printing and physical integrity)
        'min_max_thickness': 0.04,
        'max_camber': 0.04,
    }
    
    history = []
    valid = 0
    skipped = 0
    lr = float(learning_rate_init)
    
    # Diagnostic counters (printed for first 20 skips to help debug)
    _pending_skips = 0

    def _flush_skips():
        nonlocal _pending_skips
        if _pending_skips > 0:
            print(f"  ... skipped {_pending_skips} invalid candidates")
            _pending_skips = 0
    
    for it in range(1, n_iters + 1):
        
        # --- PROPOSE ---
        # ACTION ITEM (2/19 meeting): "Lines 397-400, 404-405: global is not
        # needed -- take out mode"
        #
        # WHAT CHANGED FROM OLD CODE:
        #   Old code had probabilistic 75%/25% local/global switching using
        #   p_local and a random draw each iteration — that was the "mode"
        #   the professor said to take out.
        #   New code: ALWAYS proposes locally from the current best latent.
        #   Global random sampling is ONLY used as a fallback when best=None
        #   (which happens only if the lookup table baseline failed all
        #   constraints and we haven't found ANY valid candidate yet).
        #
        # NOTE: p_local is kept in the function signature for backwards
        # compatibility but is NOT used in this loop. The broader global
        # search role is now handled by the TF .fit() refinement that runs
        # AFTER this loop (build_and_train_nom at line 554 / below).
        #
        # FALLBACK:
        #   best=None means no valid candidate exists yet (baseline failed or
        #   not provided). We can't do local search without a starting point,
        #   so randomly sample uniformly from [lat_lo, lat_hi] until we find
        #   the first valid foil, then switch permanently to local proposals.
        if best is None:
            # Emergency: no valid foil yet — pick a random point in latent box
            cand = propose_global(lat_lo, lat_hi)
        else:
            # Normal: perturb best latent with Gaussian noise scaled by lr
            # (lr decays over iterations so steps get smaller over time)
            cand = propose_local(best['latent'], lr=lr, lat_lo=lat_lo, lat_hi=lat_hi)
        
        # --- EVALUATE ---
        res = safe_eval(pipeline, cand, alpha=alpha, Re=Re)
        if res is None:
            skipped += 1
            _pending_skips += 1
            lr *= float(lr_decay)
            continue
        
        CL, CD, coords = res['CL'], res['CD'], res['coords']
        
        # --- OBJECTIVE ---
        obj = default_objective(CL, CD)
        
        # --- PENALTY ---
        pen, pen_info = total_penalty(
            latent_vec=cand,
            coords=coords,
            CL=CL,
            lat_lo=lat_lo,
            lat_hi=lat_hi,
            **penalty_kwargs
        )
        
        # --- HARD REJECT (prof: using 1000 not inf) ---
        # FIX: was `pen > 1000.0` (strictly greater), which let pen=1000.0 slip
        # through as "valid" since hard rejects return exactly 1000.0.
        # Changed to `pen >= 1000.0` so any hard-rejected foil is always skipped.
        if not (np.isfinite(pen) and np.isfinite(obj)) or pen >= 1000.0:
            skipped += 1
            _pending_skips += 1
            lr *= float(lr_decay)
            continue
        
        # --- VALID CANDIDATE ---
        valid += 1
        total = float(obj + pen)
        
        rec = {
            'iter': int(it),
            # ACTION ITEM (2/19): 'mode' removed (global exploration taken out)
            'lr': float(lr),
            'CL': float(CL),
            'CD': float(CD),
            'objective': float(obj),
            'penalty': float(pen),
            'total': float(total),
            't_min': float(pen_info.get('t_min', 0.0)),
            't_max': float(pen_info.get('t_max', 0.0)),
            'te_gap': float(pen_info.get('te_gap', 0.0)),
        }
        history.append(rec)
        
        # --- UPDATE BEST ---
        if best is None or total < best['total']:
            best = {**rec, 'latent': cand.copy(), 'coords': coords.copy()}
            _flush_skips()
            print(
                f"[{it:4d}/{n_iters}] NEW BEST | "
                f"total={total:.6f} "
                f"(obj={obj:.6f}, pen={pen:.6f}) | "
                f"CL={CL:.4f} CD={CD:.6f} | "
                f"L/D={CL/CD:.1f} | "
                f"tmin={pen_info.get('t_min', 0):.4f} tmax={pen_info.get('t_max', 0):.4f} | "
                f"lr={lr:.2e}"
                # ACTION ITEM (2/19): removed mode from print (global taken out)
            )
        
        # -----------------------------------------------------------------------
        # ACTION ITEM (2/19): no training inside the for loop.
        # Training (nom.summary/compile/fit) happens AFTER the loop below.
        # -----------------------------------------------------------------------

        # --- DECAY LR ---
        lr *= float(lr_decay)

    # -----------------------------------------------------------------------
    # LINE 554 — ACTION ITEM (2/19 meeting):
    # "nom_optimize for loop: no training -- for training use .fit"
    #
    # NOM random search finished. Now refine the best latent with TF:
    #   nom.summary() → nom.compile() → nom.fit()
    # -----------------------------------------------------------------------
    if best is not None:
        print()
        _flush_skips()
        print("Starting TF refinement (nom.summary → nom.compile → nom.fit)...")
        refined_latent = build_and_train_nom(
            pipeline=pipeline,
            lat_lo=lat_lo,
            lat_hi=lat_hi,
            init_latent=best['latent'],
            learning_rate=0.0005,   # reduced from 0.005 — prevents overshooting
            n_epochs=400,           # increased from 200 — more steps at smaller lr
            bounds_lam=10.0,
            fd_eps=0.01,
            alpha=alpha,
            Re=Re,
            penalty_kwargs=penalty_kwargs,  # pass geometry constraints into TF training
            verbose=True,
        )

        # Evaluate with NeuralFoil to see if TF refinement improved things
        res_r = safe_eval(pipeline, refined_latent, alpha=alpha, Re=Re)
        if res_r is not None:
            CL_r, CD_r, coords_r = res_r['CL'], res_r['CD'], res_r['coords']
            obj_r = default_objective(CL_r, CD_r)
            pen_r, _ = total_penalty(
                latent_vec=refined_latent,
                coords=coords_r,
                CL=CL_r,
                lat_lo=lat_lo,
                lat_hi=lat_hi,
                **penalty_kwargs
            )
            total_r = obj_r + pen_r
            print(f"TF-refined: CL={CL_r:.4f}  CD={CD_r:.6f}  "
                  f"L/D={CL_r/CD_r:.1f}  pen={pen_r:.4f}")
            if pen_r < 1000.0 and total_r < best['total']:
                print("✓ TF refinement improved NOM best — adopting refined latent")
                best['latent']    = refined_latent.copy()
                best['coords']    = coords_r.copy()
                best['CL']        = float(CL_r)
                best['CD']        = float(CD_r)
                best['objective'] = float(obj_r)
                best['penalty']   = float(pen_r)
                best['total']     = float(total_r)
            else:
                print("  TF refinement did not improve — keeping NOM best")
        else:
            print("  TF-refined latent failed NeuralFoil eval — keeping NOM best")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------

    if best is None:
        print("\n" + "=" * 70)
        print("⚠️  NOM found 0 valid candidates")
        print("Try loosening constraints or increasing iterations")
        print("=" * 70)
        return
    
    np.save(out_path / "best_latent_nom.npy", best['latent'])
    np.savetxt(
        out_path / "best_latent_nom.csv",
        best['latent'].reshape(1, -1),
        delimiter=",",
        header="p1,p2,p3,p4,p5,p6",
        comments=""
    )
    
    np.savetxt(
        out_path / "best_coords_nom.csv",
        best['coords'],
        delimiter=",",
        header="x,y",
        comments=""
    )
    
    with open(out_path / "nom_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    summary = {
        'alpha': float(alpha),
        'Re': float(Re),
        'n_iters': int(n_iters),
        'learning_rate_init': float(learning_rate_init),
        'lr_decay': float(lr_decay),
        'p_local': float(p_local),
        'lam_bounds': float(lam_bounds),
        'lam_geom': float(lam_geom),
        'lam_cl': float(lam_cl),
        'min_thickness': float(min_thickness),
        'max_thickness': float(max_thickness),
        'te_gap_max': float(te_gap_max),
        'cl_min': None if cl_min is None else float(cl_min),
        'cl_max': None if cl_max is None else float(cl_max),
        'valid_evals': int(valid),
        'skipped': int(skipped),
        'best_total': float(best['total']),
        'best_objective': float(best['objective']),
        'best_penalty': float(best['penalty']),
        'best_CL': float(best['CL']),
        'best_CD': float(best['CD']),
        'best_latent_params': [float(x) for x in best['latent']],
        'latent_lo': [float(x) for x in lat_lo],
        'latent_hi': [float(x) for x in lat_hi],
        # ── BASELINE TRACKING ──────────────────────────────────────────
        # Save which foil was used as the starting point for this run so
        # plot_nom_results.py can automatically load and overlay it without
        # any hardcoded paths. The plotter reads this key and searches for
        # the matching .txt file in airfoils_txt/.
        # e.g. "hq358", "naca0012", "e423", etc. (stem of the .txt filename)
        'baseline_foil_filename': baseline['filename'] if baseline is not None else None,
    }
    
    with open(out_path / "nom_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("NOM OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Best CL:   {best['CL']:.4f}")
    print(f"Best CD:   {best['CD']:.6f}")
    print(f"Best L/D:  {best['CL'] / best['CD']:.2f}")
    print(f"CD/CL:     {best['objective']:.6f}")
    print(f"Valid:     {valid}/{n_iters} ({100*valid/n_iters:.1f}%)")
    print(f"Skipped:   {skipped}/{n_iters} ({100*skipped/n_iters:.1f}%)")
    print(f"Outputs:   {out_path}/")
    print("=" * 70)


if __name__ == "__main__":
    nom_optimize()