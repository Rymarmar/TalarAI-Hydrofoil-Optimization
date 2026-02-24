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


class LatentLayer(tf.keras.layers.Layer):
    """
    A Layer that holds the 6 trainable latent weights.

    WHY A LAYER INSTEAD OF tf.Variable DIRECTLY:
      Keras only tracks variables as trainable if they are registered through
      a Layer (via add_weight) or a keras.Model subclass. A bare tf.Variable
      assigned as self.latent_weights = tf.Variable(...) is NOT automatically
      found by model.trainable_variables, which causes the
      'not enough values to unpack' error during nom.fit().

      Using add_weight() properly registers the 6 params so:
        - nom.summary() shows 6 Trainable params
        - nom.fit() finds them and passes them to Adam
        - GradientTape tracks them correctly
    """

    def __init__(self, lat_lo: np.ndarray, lat_hi: np.ndarray,
                 init_latent: np.ndarray | None = None, **kwargs):
        super().__init__(**kwargs)
        if init_latent is not None:
            init = np.array(init_latent, dtype=np.float32).reshape(6)
        else:
            init = np.random.uniform(lat_lo, lat_hi).astype(np.float32)

        # add_weight registers this as a proper Keras trainable variable
        # so nom.summary() and nom.fit() both see exactly 6 trainable params
        self.latent_weights = self.add_weight(
            name="latent_weights",
            shape=(6,),
            initializer=tf.constant_initializer(init),
            trainable=True,
        )
        self._lat_lo = tf.constant(lat_lo.astype(np.float32))
        self._lat_hi = tf.constant(lat_hi.astype(np.float32))

    def call(self, inputs=None):
        # Return the 6 latent weights as shape (1, 6) for the decoder.
        #
        # WHY WE ACCEPT AND IGNORE `inputs`:
        #   When nom.fit() runs, Keras calls this layer's call() method and
        #   passes the actual batch tensor from dummy_dataset as `inputs`.
        #   We do NOT use that batch data — our 6 latent weights are the
        #   things being optimized, not the incoming data.
        #   Accepting `inputs` here (even though we ignore it) prevents Keras
        #   from raising "call() missing required argument 'inputs'" errors
        #   during .fit() and .predict() calls.
        _ = inputs  # intentionally unused — this layer has no data input
        return tf.expand_dims(self.latent_weights, axis=0)   # shape (1, 6) for decoder

    def bounds_penalty(self) -> tf.Tensor:
        """
        ACTION ITEM: "Another penalty for over/under min/max 6 params"
        Differentiable ReLU penalty for latent params outside [lat_lo, lat_hi].
        """
        z = self.latent_weights                          # shape (6,)
        below = tf.nn.relu(self._lat_lo - z)
        above = tf.nn.relu(z - self._lat_hi)
        return tf.reduce_sum(below + above)


class NOMTrainingModel(tf.keras.Model):
    """
    ACTION ITEM (2/19): Trainable NOM model using nom.summary/compile/fit.

    STRUCTURE (matches professor's whiteboard exactly):
      Trainable:  latent_layer  →  6 weights p1..p6, via add_weight()
      Frozen:     decoder       →  Dense 6→100→1000→80, trainable=False
      Output:     y_pred (1, 80) — decoded foil coordinates

    LOSS (professor's diagram: loss = CD/CL + penalty):
      NeuralFoil is a Python function, not a TF layer, so gradients can't
      flow through it via autograd. We use tf.py_function() to call it
      INSIDE train_step, then use finite differences to estimate the gradient
      of CD/CL w.r.t. the 6 latent weights. Adam then updates those 6 weights.

      This keeps the full .fit() API while using the real CD/CL objective.

    WHY tf.py_function:
      tf.py_function wraps any Python function so it can be called inside
      TF's execution graph. It returns a tensor but TF knows not to try
      to differentiate through it — we handle the gradient ourselves via
      finite differences.
    """

    def __init__(self, decoder_model: tf.keras.Model,
                 lat_lo: np.ndarray, lat_hi: np.ndarray,
                 init_latent: np.ndarray | None = None,
                 bounds_lam: float = 10.0,
                 fd_eps: float = 0.01,
                 neuralfoil_fn=None,    # callable: latent (6,) → (CL, CD)
                 ):
        super().__init__()

        # ACTION ITEM: "Have everything set to be non-trainable / freeze"
        decoder_model.trainable = False
        self.decoder    = decoder_model
        self._lat_lo    = tf.constant(lat_lo.astype(np.float32))
        self._lat_hi    = tf.constant(lat_hi.astype(np.float32))
        self.bounds_lam = tf.constant(float(bounds_lam), dtype=tf.float32)
        self.fd_eps     = float(fd_eps)
        self._nf_fn     = neuralfoil_fn   # stored for use in train_step

        # ACTION ITEM: 6 trainable latent weights via add_weight() so
        # nom.summary() shows Trainable params: 6, and nom.fit() finds them
        self.latent_layer = LatentLayer(lat_lo, lat_hi, init_latent,
                                        name="latent_layer")

    def call(self, inputs=None, training=False):
        """
        Forward pass: 6 latent weights → frozen decoder → (1,80) y-coords.

        WHY WE PASS `inputs` THROUGH TO latent_layer:
          When nom.fit() runs, Keras calls this model's call() with the
          actual batch from dummy_dataset. We don't use the batch data, but
          we must pass it down to latent_layer so Keras's internal graph
          tracing doesn't error. latent_layer.call() accepts and ignores it.

        TRAINING=FALSE ON DECODER:
          The decoder is frozen (decoder.trainable = False set in __init__).
          Passing training=False also disables any Dropout/BatchNorm layers
          inside the decoder that might otherwise behave differently at
          inference vs training time. This ensures the decoder output is
          stable and deterministic during NOM's gradient estimation.
        """
        z = self.latent_layer(inputs)              # passes dummy input, returns (1, 6)
        return self.decoder(z, training=False)     # frozen decoder: (1, 6) → (1, 80)

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
          called on the current 6 latent weights, not from any training data.

        LOSS = CD/CL (from NeuralFoil) + bounds_penalty
        GRADIENT = finite differences across the 6 latent dimensions

        WHY WE CAN'T CALL .numpy() DIRECTLY HERE:
          When Keras runs .fit(), train_step is traced and executed inside a
          TF graph context (not eager mode). In graph mode, tf.Variable and
          tf.Tensor objects are symbolic — they don't have actual numeric
          values yet, so calling .numpy() raises:
            NotImplementedError: numpy() is only available when eager execution is enabled.

        THE FIX — tf.py_function:
          tf.py_function(func, inputs, Tout) wraps a plain Python function so
          it can be called INSIDE the TF graph. When TF reaches a py_function
          node during execution, it:
            1. Temporarily switches to eager mode for that call only
            2. Runs func with real numpy-convertible tensors as inputs
            3. Returns the result as a tf.Tensor back to the graph
          This lets our Python/NumPy NeuralFoil code run safely inside .fit().

        HOW IT WORKS HERE:
          We define _compute_grads_and_loss(z_tensor) as a Python function
          that accepts the latent weights as a tensor, calls .numpy() on it
          (safe inside py_function), runs NeuralFoil + finite differences,
          and returns [loss, grad_0, ..., grad_5] as a flat list of floats.
          tf.py_function wraps this and hands back TF tensors we can use
          with optimizer.apply_gradients().
        """
        _ = data  # dummy tensor from dummy_dataset — intentionally unused

        def _compute_grads_and_loss(z_tensor):
            """
            Pure Python function that runs inside tf.py_function (eager mode).
            Receives the current latent weights as a numpy-convertible tensor.
            Returns a flat tf.float32 tensor: [loss_total, g0, g1, g2, g3, g4, g5]
            (1 loss value + 6 gradient values = 7 floats total).
            """
            # .numpy() is safe here because tf.py_function gives us real values
            z_np  = z_tensor.numpy().astype(np.float64)   # shape (6,)
            lo_np = self._lat_lo.numpy().astype(np.float64)
            hi_np = self._lat_hi.numpy().astype(np.float64)
            lam   = float(self.bounds_lam.numpy())

            # --- Evaluate CD/CL at the CURRENT latent (baseline for finite-diff) ---
            # This is the aerodynamic objective we are minimizing.
            loss_c = self._eval_cd_over_cl(z_np)

            # --- Bounds penalty: penalize each latent dim that left [lo, hi] ---
            # relu(lo - z) > 0 when z went below its minimum training value
            # relu(z - hi) > 0 when z went above its maximum training value
            below = np.maximum(0.0, lo_np - z_np)
            above = np.maximum(0.0, z_np - hi_np)
            bp = lam * float(np.sum(below + above))

            if not np.isfinite(loss_c):
                # The current latent produces an invalid foil (CL<=0 or NaN).
                # We cannot compute a meaningful CD/CL gradient from here.
                # Return a large loss and zero gradients so Adam holds position.
                # The bounds penalty gradient (computed below) will still steer
                # the latent back toward valid parameter space if it drifted out.
                loss_total = 1e6 + bp
                grads_np = np.zeros(6, dtype=np.float32)
                # Still add bounds gradient so Adam can recover from out-of-bounds
                grads_np += np.where(z_np < lo_np, -lam,
                            np.where(z_np > hi_np,  lam, 0.0)).astype(np.float32)
                result = np.array([loss_total] + grads_np.tolist(), dtype=np.float32)
                return tf.constant(result, dtype=tf.float32)

            loss_total_val = loss_c + bp

            # --- Finite-difference gradient of CD/CL w.r.t. each of 6 latent dims ---
            # For each dimension i:
            #   z_plus = current z with z[i] bumped up by fd_eps
            #   grad[i] ≈ (CD/CL(z_plus) - CD/CL(z_current)) / fd_eps
            # This is a forward finite difference: slope = rise / run.
            # Adam uses these 6 slope values to decide which direction and
            # how far to move each latent weight to reduce CD/CL.
            # Cost: 6 extra NeuralFoil calls per epoch (one per latent dim).
            grads_np = np.zeros(6, dtype=np.float32)
            for i in range(6):
                z_plus = z_np.copy()
                z_plus[i] += self.fd_eps          # bump dimension i up by epsilon
                loss_p = self._eval_cd_over_cl(z_plus)
                if np.isfinite(loss_p):
                    # Forward finite difference: (f(z+eps) - f(z)) / eps
                    grads_np[i] = float((loss_p - loss_c) / self.fd_eps)
                # If loss_p is inf (this perturbation broke the foil geometry),
                # leave grad[i] = 0 so Adam doesn't push in that direction.

            # --- Add analytical gradient of bounds penalty ---
            # The bounds penalty is: lam * sum(relu(lo-z) + relu(z-hi))
            # Its gradient w.r.t. z[i]:
            #   -lam  when z[i] < lo[i]  (pull z up toward lo)
            #   +lam  when z[i] > hi[i]  (push z down toward hi)
            #    0    otherwise           (inside bounds, no contribution)
            grads_np += np.where(z_np < lo_np, -lam,
                        np.where(z_np > hi_np,  lam, 0.0)).astype(np.float32)

            # Pack loss + 6 gradients into a single flat float32 tensor.
            # tf.py_function requires all outputs to be TF tensors.
            # We unpack this in the calling code below.
            result = np.array([loss_total_val] + grads_np.tolist(), dtype=np.float32)
            return tf.constant(result, dtype=tf.float32)

        # --- Call the Python function through tf.py_function ---
        # This is the bridge between TF's graph execution and our Python/NumPy code.
        # Tout=[tf.float32] tells TF the function returns one float32 tensor.
        # The output is shape (7,): [loss, g0, g1, g2, g3, g4, g5].
        result_tensor = tf.py_function(
            func=_compute_grads_and_loss,
            inp=[self.latent_layer.latent_weights],   # pass latent as TF tensor
            Tout=[tf.float32],                         # expect one float32 tensor back
        )[0]  # [0] because Tout is a list — unwrap the single output

        # Unpack the 7-element result tensor back into loss and gradients.
        # result_tensor[0]   = loss_total (scalar)
        # result_tensor[1:7] = grad for each of the 6 latent dims
        loss_total_t = result_tensor[0:1]          # shape (1,) — Keras expects a tensor
        grads_t      = result_tensor[1:7]          # shape (6,) — one grad per latent dim

        # Apply the 6 gradients to the 6 latent weights via Adam.
        # Adam wraps each gradient with per-dimension momentum/variance estimates,
        # so the effective step size adapts automatically for each latent param.
        self.optimizer.apply_gradients(
            [(grads_t, self.latent_layer.latent_weights)]
        )

        # Return metrics dict. Keras logs these and prints them during .fit().
        # loss_total_t[0] extracts the scalar from the (1,) tensor for display.
        return {
            "loss":  loss_total_t[0],
            "CD_CL": result_tensor[0],
        }

    def get_latent_numpy(self) -> np.ndarray:
        """
        Return current latent weights as a numpy array of shape (6,).

        WHY .numpy() IS SAFE HERE:
          This method is called OUTSIDE of nom.fit() — after fit() completes,
          TF returns to eager mode. .numpy() only crashes when called inside
          train_step, because fit() traces train_step into a TF graph (non-eager).
          Everything outside fit() runs eagerly, so .numpy() works normally here.
        """
        return self.latent_layer.latent_weights.numpy().reshape(6)


def build_and_train_nom(
    pipeline,
    lat_lo: np.ndarray,
    lat_hi: np.ndarray,
    *,
    init_latent: np.ndarray | None = None,
    learning_rate: float = 0.005,
    n_epochs: int = 200,
    bounds_lam: float = 10.0,
    fd_eps: float = 0.01,
    alpha: float = 1.0,
    Re: float = 450000.0,
    verbose: bool = True,
) -> np.ndarray:
    """
    ACTION ITEM (2/19 meeting): nom.summary() → nom.compile() → nom.fit()

    PROFESSOR'S EXACT SEQUENCE:
      1. nom = NOMTrainingModel(...)
      2. nom.summary()                    verify 6 trainable, rest frozen
      3. nom.compile(Adam(lr))            set optimizer
      4. nom.fit(data, epochs=n_epochs)   run gradient descent via .fit()

    LOSS inside .fit():
      Each epoch, train_step calls NeuralFoil to get real CD/CL,
      estimates the gradient via finite differences across the 6 latent dims,
      and applies it with Adam. This is the professor's diagram implemented
      fully using .fit().

    OUTPUT: best refined latent shape (6,)
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
        print(f"  Trainable params:     {n_train}   ← should be exactly 6 (p1..p6)")
        print(f"  Non-trainable params: {n_frozen}  ← decoder frozen")
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
        print(f"  Loss = CD/CL (NeuralFoil) + bounds_penalty")
        print(f"  Gradient: finite differences, eps={fd_eps}")
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
    n_iters: int = 3000,
    
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
            learning_rate=0.005,
            n_epochs=200,
            bounds_lam=10.0,
            fd_eps=0.01,
            alpha=alpha,
            Re=Re,
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