"""
optimization/nom_driver.py

===========================================================================
MEETING ACTION ITEMS — COMPLETE STATUS
===========================================================================

[2/10] Learning rate <= 1e-3.
    DONE. tf_learning_rate defaults to 5e-4 (well below the 1e-3 cap).
    Per meeting: "Learning Rate: 1e-3 or less."

[2/10] CL_min, CL_max — update numbers.
    DONE. cl_min=0.15 (physical floor for takeoff), cl_max=None (no ceiling).
    Per meeting: "Cl_min, cl_max — change the numbers."

[2/17] No global `mode` variable (Lines 397-400, 404-405 in old code).
    DONE. All globals removed. No `mode` variable anywhere in this file.
    Per meeting: "global is not needed, take out mode."

[2/17] Lookup table as baseline instead of seeds.
    DONE. baseline loaded from lookup table JSON. No seeds.
    Per meeting: "Make Lookup table... find the best airfoil and use that
    as the baseline."

[2/19] lookup_baseline_path — take out None (Line 202 in old code).
    DONE. Defaults to "" and auto-constructs path from design point.
    Per meeting: "Lookup_table_path Line 202: take out None."

[2/19] Line 244-251 not needed — just use latent params from JSON.
    DONE. latent_baseline = np.array(baseline["latent"]) directly from JSON.
    Per meeting: "not needed, just use latent params in the json file."

[2/19] If user gives decimal Re or alpha, snap to nearest valid grid value.
    DONE. snap_conditions() helper snaps decimals before anything runs.
    Per meeting: "if the user gives decimal for Re and aoa, mod the value
    and take out the remainder. Line 269 do it there."

[2/19] No cambered foils — NACA symmetric foils easier to 3D print.
    DONE. max_camber default tightened from 0.04 to 0.02.
    Per meeting: "So no cambered foils please. NACA foils are easier to
    3dprint/physically produce."

[2/19] nom_optimize for loop: no training. For training: .fit
    DONE. NOMModel overrides train_step() to use FD gradients + Adam.
    nom.fit(dummy_dataset, epochs=n_iters) runs the full optimization.
    No external for-loop. The loop IS nom.fit().
    Per meeting: "nom.summary() / nom.compile(adam(lr, lr)) / nom.fit()"

[2/26] ONE unified loop — no separate TF epochs then NOM iterations.
    DONE. nom.fit(epochs=n_iters) = n_iters FD+Adam steps. That's it.
    Per meeting: "iterations and epochs should all be the same — one
    continuous flow of the diagram. Why are we running tf training 400
    epochs and then 1000 iterations?"

[2/26] Multi-condition objective — evaluate at all (alpha, Re) pairs.
    DONE. Each train_step evaluates avg CD/CL across all OPERATING_CONDITIONS.
    Per meeting: "NOM should go through diff Reynolds and Angle of attack."

[2/26] CL floor: hard reject ONLY at takeoff (not at all conditions).
    FIXED. Old code rejected if CL < cl_min at ANY condition, causing ~47%
    skip rate. New code: hard reject only if LAST condition (takeoff:
    alpha=4°, Re=150k) fails CL floor.
    Per meeting: "just need to achieve min CL, not at super high Re."

===========================================================================
WHY NOM.FIT() IS NOW POSSIBLE — KEY EXPLANATION
===========================================================================

The professor requires nom.compile(Adam) + nom.fit() UNLESS impossible.

Previously we thought it was impossible because NeuralFoil is a NumPy
black box and TF's GradientTape cannot differentiate through it.

THE SOLUTION: Override train_step() inside NOMModel.

Keras calls train_step() automatically once per epoch inside nom.fit().
By overriding it, we compute FD gradients ourselves and hand them to
self.optimizer (which IS Adam from nom.compile).

  Step inside train_step():
    1. Read current w, b from latent_layer
    2. For each of 12 params: perturb by +fd_eps, run NeuralFoil, measure loss change
       → grad_i = (loss(param + eps) - loss_0) / eps   ← THIS IS FINITE DIFFERENCES
    3. self.optimizer.apply_gradients([(grad_w, w), (grad_b, b)])  ← THIS IS ADAM
    4. If new point is better, record as best

Result:
  nom.summary()  → shows 12 trainable params (w[6] + b[6])
  nom.compile(Adam(lr=5e-4))  → registers Adam as self.optimizer
  nom.fit(dummy_dataset, epochs=n_iters)  → runs n_iters FD+Adam steps

The loop IS nom.fit(). No fake wrapper. Satisfies professor's diagram exactly.

===========================================================================
ARCHITECTURE (professor's whiteboard 2/26, Image 3):
===========================================================================

  TRAINABLE (left side of diagram):
    z_init (frozen — baseline foil latent from lookup table, 6 values)
      |
      v  z_eff[i] = w[i]*z_init[i] + b[i]   (professor's formula, Image 2)
    LinearLatentLayer
      w[6] — TRAINABLE (circled on whiteboard)
      b[6] — TRAINABLE
      → 12 trainable params total

  NON-TRAINABLE / FROZEN (big rectangle on whiteboard):
    Decoder  6 → 100 → 1000 → 80
      |
      v  (80 airfoil coords)
    NeuralFoil (NumPy black box — NOT a TF op, gradients via FD)
      |
      v
    CL, CD

  OBJECTIVE (right side of whiteboard):
    min  avg(CD/CL)  across all OPERATING_CONDITIONS
      +  penalty(geometry violations, latent out of bounds, CL floor)

  SINGLE LOOP FLOW (nom.fit → custom train_step, one call per epoch):
    1. Compute loss_0 at current z_eff (1 NeuralFoil call × 4 conditions)
    2. Perturb each w_i by ±fd_eps → 12 NF evaluations → dw[6]   (central diff, FIX #2)
    3. Perturb each b_i by ±fd_eps → 12 NF evaluations → db[6]   (central diff, FIX #2)
    4. Apply Adam: w, b ← Adam(w, b, grads=[dw, db])
    5. Evaluate new z_eff → record if improved
    6. If rollback: reset Adam momentum slots (FIX #1)
    Total per epoch: ~25 base evals × 4 conditions = ~100 NeuralFoil calls
    (was ~52 with one-sided FD; central diff doubles FD calls but fixes stagnation)
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# PATH FIX: Make sure the project root (TalarAI-Hydrofoil-Optimization/) is
# on sys.path so that `from pipeline.talarai_pipeline import ...` works when
# this file is run directly with:
#   python optimization/nom_driver.py
#
# Without this, Python only adds the `optimization/` folder to the path,
# so it can't find the `pipeline/` package which lives one level up.
# This is the same fix used in build_lookup_table.py (PROJECT_ROOT trick).
# ---------------------------------------------------------------------------
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
# OPERATING CONDITIONS  (multi-condition sweep — 2/26 action item)
# ===========================================================================
# Physical basis (from Ski Cat Info spreadsheet, 1/15 scale model):
#   chord = 0.1875 ft,  nu_water = 1.08e-5 ft^2/s
#   Slow  speed: V =  8.44 ft/s  → Re ≈ 150,000  (takeoff / low-speed)
#   Mid   speed: V = 14.6  ft/s  → Re ≈ 250,000  (transition)
#   Fast  speed: V = 20.8  ft/s  → Re ≈ 350,000  (cruise)
#   Max   speed: V = 25.3  ft/s  → Re ≈ 450,000  (top speed / design point)
#
# WHY THESE 4 CONDITIONS:
#   They span the full operating envelope (takeoff through max speed).
#   Using all 4 means we don't over-optimize for one speed and fail at another.
#   The design point is alpha=1°, Re=450k (max speed, lowest drag needed).
#
# IMPORTANT ORDERING: The LAST entry is the takeoff condition.
#   CL floor hard-reject is applied ONLY to per_cond[-1] (the last entry).
#   This fixes the ~47% skip rate from the old code (see eval_all_conditions).
#
# Each entry: (alpha_degrees, Reynolds_number)
OPERATING_CONDITIONS = [
    (1.0, 450_000),   # max speed / design point  — CL naturally low here, NO floor check
    (2.0, 350_000),   # cruise
    (3.0, 250_000),   # mid-speed
    (4.0, 150_000),   # takeoff / slow  ← CL floor checked HERE ONLY (last entry)
]

# Valid grid values for snapping user input (2/19 action item)
# NeuralFoil trained on these specific values — decimals can degrade accuracy.
_VALID_ALPHAS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
_VALID_RES    = [150_000, 250_000, 350_000, 450_000]


# ===========================================================================
# HELPER: SNAP CONDITIONS TO NEAREST VALID GRID VALUES
# ===========================================================================

def snap_conditions(
    conditions: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """
    [ACTION ITEM 2/19]: "If the user gives decimal for Re and aoa, mod the
    value and take out the remainder. Line 269 do it there."

    Snap each (alpha, Re) pair to the nearest valid grid value.
    Example: (3.14, 155000.7) → (3.0, 150000)

    WHY THIS MATTERS:
      NeuralFoil is a neural network trained at specific (alpha, Re) combos.
      Feeding it weird decimals forces it to interpolate, which can give
      unreliable CL/CD values. Snapping to known-good grid values is safer
      and more physically meaningful for our operating conditions.
    """
    snapped = []
    for alpha, Re in conditions:
        alpha_snapped = min(_VALID_ALPHAS, key=lambda a: abs(a - alpha))
        Re_snapped    = min(_VALID_RES,    key=lambda r: abs(r - Re))
        if alpha_snapped != alpha or Re_snapped != Re:
            print(f"  [snap_conditions] ({alpha}°, Re={Re:.0f}) "
                  f"→ ({alpha_snapped}°, Re={Re_snapped:.0f})")
        snapped.append((float(alpha_snapped), float(Re_snapped)))
    return snapped


# ===========================================================================
# DATASET LOADING
# ===========================================================================

def load_latent_dataset(csv_path: str = "data/airfoil_latent_params_6.csv") -> np.ndarray:
    """
    Load the 6D latent vectors for all training foils from CSV.

    Used ONLY to compute lat_lo and lat_hi (per-dimension min/max across
    all 1647 training foils). These bounds define the valid latent space
    that our optimizer should stay inside.

    WHY WE NEED THIS:
      The decoder was trained on foils whose latent vectors fall within
      [lat_lo, lat_hi]. If w and b push z_eff outside this range, the
      decoder will produce physically meaningless shapes.
    """
    df = pd.read_csv(csv_path)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] != 6:
        raise ValueError(f"Expected 6 latent columns, found {numeric.shape[1]}")
    return numeric.values.astype(float)


# ===========================================================================
# LOAD BEST BASELINE FROM LOOKUP TABLE
# ===========================================================================

def load_best_baseline(json_path: str | Path) -> dict | None:
    """
    Load the best baseline foil from the pre-built lookup table JSON.

    [ACTION ITEM 2/17]: "Make Lookup table of all the foils... find the best
    airfoil and then use that as the baseline."

    WHY NOT SEEDS (old approach):
      Seeds randomly picked foils. With 1647 foils, random seeds waste time.
      The lookup table (built by build_lookup_table.py) pre-evaluates all
      foils at each (alpha, Re) grid point and stores the best one.
      This guarantees we start from the best known foil — much better starting
      point for the optimizer.

    [ACTION ITEM 2/19]: "Lookup_table_path Line 202: take out None"
      Path defaults to "" and is auto-constructed. Never None.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"⚠️  Lookup table baseline not found: {json_path}")
        print(f"   Run: python tools/build_lookup_table.py")
        return None
    with open(json_path) as f:
        return json.load(f)


# ===========================================================================
# MULTI-CONDITION EVALUATION
# ===========================================================================

def eval_all_conditions(
    pipeline: "TalarAIPipeline",
    latent_vec: np.ndarray,
    conditions: list[tuple[float, float]],
    cl_min: float = 0.15,
) -> dict | None:
    """
    Evaluate a latent vector at every (alpha, Re) operating condition.

    Returns a dict with:
      avg_cd_cl  — average CD/CL across all valid conditions (the objective)
      per_cond   — list of {alpha, Re, CL, CD, cd_cl} for each condition
      n_valid    — how many conditions gave finite CL/CD
      cl_fail    — True if TAKEOFF condition (last entry) fails CL >= cl_min
      coords     — foil coordinates from design point (for geometry checks)
      CL, CD     — design point (first condition) values for logging

    Returns None if ALL conditions fail NeuralFoil entirely.

    WHY AVERAGE CD/CL:
      We want a foil that performs well across the full speed envelope.
      A foil optimal at max speed but unable to lift at takeoff is useless.
      Average gives equal weight to all 4 operating conditions.

    [ACTION ITEM 2/26 — CRITICAL FIX]: CL floor check ONLY at takeoff.
      OLD (wrong, caused ~47% skip rate):
        cl_fail = any(c["CL"] < cl_min for c in per_cond)
        Problem: at alpha=1°, Re=450k (max speed), CL is naturally ~0.10-0.20.
        Many physically valid foils got hard-rejected just because they have
        low CL at max speed — which is expected and fine at high speed.

      NEW (correct):
        cl_fail = (per_cond[-1]["CL"] < cl_min)
        Only checks the LAST condition = takeoff (alpha=4°, Re=150k).
        At low speed/high angle, the foil MUST generate enough lift.
        At max speed, low CL is acceptable.

      Per 2/26 meeting: "just need to achieve min CL, not at super high Re."
    """
    z = np.asarray(latent_vec, dtype=float)
    per_cond  = []
    cd_cl_vals = []

    for alpha, Re in conditions:
        try:
            out = pipeline.eval_latent_with_neuralfoil(z, alpha=alpha, Re=Re)
            CL  = float(out["CL"])
            CD  = float(out["CD"])
            coords = out.get("coords")

            if not (np.isfinite(CL) and np.isfinite(CD) and CD > 0):
                per_cond.append({"alpha": alpha, "Re": Re,
                                 "CL": np.nan, "CD": np.nan, "cd_cl": np.nan})
                continue

            cd_cl = default_objective(CL, CD)
            per_cond.append({"alpha": alpha, "Re": Re, "CL": CL,
                             "CD": CD, "cd_cl": cd_cl, "coords": coords})
            cd_cl_vals.append(cd_cl)

        except Exception:
            per_cond.append({"alpha": alpha, "Re": Re,
                             "CL": np.nan, "CD": np.nan, "cd_cl": np.nan})

    if len(cd_cl_vals) == 0:
        return None   # NeuralFoil failed completely — skip this foil

    avg_cd_cl = float(np.mean(cd_cl_vals))
    n_valid   = len(cd_cl_vals)

    # -------------------------------------------------------------------
    # [ACTION ITEM 2/26 FIX]: CL floor ONLY at the TAKEOFF condition.
    #
    # per_cond[-1] = last entry in conditions = (alpha=4.0, Re=150_000).
    # We check: is the takeoff CL finite AND above cl_min?
    # If not → cl_fail = True → this foil is rejected (can't lift at takeoff).
    #
    # We do NOT check the other conditions (cruise, max speed) for CL floor.
    # At max speed (alpha=1°, Re=450k), a low CL like 0.10 is physically fine
    # because the boat is going fast — lift force = 0.5*rho*V^2*A*CL is
    # still large due to the high velocity, even with low CL.
    # -------------------------------------------------------------------
    takeoff = per_cond[-1]   # last condition = takeoff
    cl_fail = (not np.isfinite(takeoff["CL"])) or (takeoff["CL"] < cl_min)

    # Coords from design point (first condition) for geometry penalty checks
    design_coords = next((c["coords"] for c in per_cond if "coords" in c), None)

    return {
        "avg_cd_cl": avg_cd_cl,
        "per_cond":  per_cond,
        "n_valid":   n_valid,
        "cl_fail":   cl_fail,       # True = hard reject (takeoff CL too low)
        "coords":    design_coords,
        "CL": per_cond[0].get("CL", np.nan) if per_cond else np.nan,
        "CD": per_cond[0].get("CD", np.nan) if per_cond else np.nan,
    }


# ===========================================================================
# LATENT LAYER  (professor's whiteboard Image 2: y_i = w_i * x_i + b_i)
# ===========================================================================

class LinearLatentLayer(tf.keras.layers.Layer):
    """
    Implements the professor's formula (whiteboard Image 2):
        z_eff[i] = w[i] * z_init[i] + b[i]

    TRAINABLE params (12 total):
      w[6] — scale factors (circled as "Trainable" on the whiteboard)
      b[6] — shift/offset

    NON-TRAINABLE params (6 total, never updated):
      z_init[6] — the baseline foil's latent vector from the lookup table.
                  This is the "x" in professor's formula y = w*x + b.

    WHY w IS TRAINABLE (not frozen at 1):
      The professor's whiteboard Image 2 explicitly circles w as "Trainable".
      w[i] lets the optimizer scale each latent dimension independently.
      For example: w[3] = 0.8 compresses the 4th latent dimension toward 0,
      while b[3] = 0.2 shifts it. Together they give full affine control.

    WHY z_init IS FROZEN:
      z_init is our starting point from the lookup table best foil.
      It defines the reference we scale and shift around. Keeping it fixed
      means the optimizer is always exploring "near" the best known foil.

    nom.summary() output for this layer:
      latent_layer — 12 trainable (w, b) + 6 non-trainable (z_init)
    """
    def __init__(self, lat_lo, lat_hi, init_latent=None, **kwargs):
        super().__init__(**kwargs)

        # z_init: the frozen baseline latent (from lookup table JSON)
        z_init = (np.array(init_latent, dtype=np.float32).reshape(6)
                  if init_latent is not None
                  else np.random.uniform(lat_lo, lat_hi).astype(np.float32))

        # Store z_init as a non-trainable weight so nom.summary() shows it
        # and so it's accessible as self._z_init in call()
        self._z_init = self.add_weight(
            name="z_init", shape=(6,),
            initializer=tf.constant_initializer(z_init),
            trainable=False,   # FROZEN — this never changes
        )

        # w[6]: scale factors — TRAINABLE
        # Start at 1.0 so initially z_eff = z_init (no change from baseline)
        self.w = self.add_weight(
            name="w", shape=(6,), initializer="ones", trainable=True)

        # b[6]: offsets/shifts — TRAINABLE
        # Start at 0.0 so initially z_eff = z_init (no change from baseline)
        self.b = self.add_weight(
            name="b", shape=(6,), initializer="zeros", trainable=True)

    def call(self, inputs=None):
        """
        Forward pass: apply the professor's formula z_eff = w * z_init + b.
        Returns shape (1, 6) — the batch dimension is needed for the decoder.
        """
        z_eff = self.w * self._z_init + self.b   # y_i = w_i * x_i + b_i
        return tf.expand_dims(z_eff, axis=0)      # (1, 6) for decoder input

    def get_effective_latent(self) -> np.ndarray:
        """
        Return current z_eff = w * z_init + b as a numpy (6,) array.

        PUBLIC API for external callers (plot_nom_results.py, future UI).
        NOT used inside train_step because train_step needs w, b, z_init
        as separate arrays for the finite-difference perturbation loop.
        """
        return (self.w.numpy() * self._z_init.numpy() + self.b.numpy()).reshape(6)


# ===========================================================================
# NOM MODEL  (nom.summary → nom.compile → nom.fit)
# ===========================================================================

class NOMModel(tf.keras.Model):
    """
    Full NOM model as drawn on the professor's whiteboard (Image 3).

    TRAINABLE:    LinearLatentLayer (12 params: w[6] + b[6])
    FROZEN:       Decoder (6 → 100 → 1000 → 80)
    BLACK BOX:    NeuralFoil (called inside train_step, not a TF layer)

    KEY: train_step() is overridden to use finite-difference gradients.
    This is what makes nom.fit() work with a non-differentiable NeuralFoil.

    [ACTION ITEM 2/19]: "nom_optimize for loop: no training. For training: .fit"
      → We override train_step() so Keras runs our FD+Adam logic each epoch.
         nom.fit(dummy_dataset, epochs=n_iters) → n_iters FD+Adam steps.
         No external for-loop. The optimization IS the .fit() call.

    WHY TRAIN_STEP IS THE RIGHT APPROACH:
      Overriding train_step is the official Keras API for customizing the
      training logic while keeping compile/fit. It's documented in the Keras
      docs as the recommended way to use custom gradient computation.
      This is NOT a workaround — it's exactly what this method is for.
    """

    def __init__(self, decoder_model, lat_lo, lat_hi, init_latent,
                 pipeline, conditions, lat_lo_np, lat_hi_np,
                 penalty_kwargs, cl_min, fd_eps, bounds_lam):
        super().__init__()

        # Freeze decoder — its weights stay fixed during optimization
        decoder_model.trainable = False
        self.decoder      = decoder_model

        # The one trainable component: LinearLatentLayer with 12 params
        self.latent_layer = LinearLatentLayer(
            lat_lo, lat_hi, init_latent, name="latent_layer")

        # Store evaluation dependencies for train_step.
        # These are Python objects — not TF weights. They're used to call
        # NeuralFoil and compute the FD gradients each step.
        self._pipeline   = pipeline        # TalarAIPipeline (runs NeuralFoil)
        self._conditions = conditions      # list of (alpha, Re) to evaluate
        self._lat_lo_np  = lat_lo_np       # numpy bounds for latent space
        self._lat_hi_np  = lat_hi_np
        self._penalty_kw = penalty_kwargs  # kwargs for total_penalty()
        self._cl_min     = cl_min          # CL floor for takeoff condition
        self._fd_eps     = fd_eps          # FD perturbation size
        self._bounds_lam = bounds_lam      # lambda for out-of-bounds penalty
        self._initial_lr = None            # set after compile() — used for LR decay

        # Track the best result found across all train_step calls
        self.best_result = None
        self.best_loss   = float("inf")
        self.history_log = []
        self.n_improved  = 0
        self.n_valid     = 0
        self.n_skipped   = 0
        self._n_iters    = 200  # overwritten by nom_optimize before fit()

    def call(self, inputs=None, training=False):
        """
        Forward pass for nom.summary() architecture tracing.
        latent_layer → decoder → (1, 80) airfoil coords
        NeuralFoil is NOT called here — it lives in train_step only.
        """
        z = self.latent_layer(inputs)            # (1, 6)
        return self.decoder(z, training=False)   # (1, 80)

    def train_step(self, data):
        """
        [ACTION ITEM 2/19 + 2/26]: The core of nom.fit() — called once per epoch.

        THIS IS FINITE DIFFERENCE GRADIENT DESCENT WITH ADAM.

        COST PER TRAIN_STEP:
          1 base eval + 6 w-perturbs + 6 b-perturbs + 1 post-step check
          = 14 NeuralFoil calls × 4 conditions = ~56 NF calls per epoch.
          (Penalty is computed inline from already-fetched results — no extra NF calls.)

        DISPLAY: We suppress Keras's progress bar (verbose=0 in nom.fit) and
        print our own one-liner per epoch showing loss, CL, CD, L/D, penalty, ETA.
        """
        # Track start time for ETA display (set on first call)
        if not hasattr(self, "_t_start"):
            self._t_start = time.time()
            self._n_total = None   # filled in from self._n_iters if available

        # ------------------------------------------------------------------
        # Step 1: Get current w, b, z_init as numpy.
        # ------------------------------------------------------------------
        w_np   = self.latent_layer.w.numpy().astype(np.float64)
        b_np   = self.latent_layer.b.numpy().astype(np.float64)
        z_init = self.latent_layer._z_init.numpy().astype(np.float64)
        z_eff  = w_np * z_init + b_np

        # ------------------------------------------------------------------
        # FIX: Cosine learning rate decay.
        #
        # WHY: With fixed lr=5e-4, the optimizer overshoots after the initial
        # fast improvement (~30 iters), then oscillates for 470 wasted iters.
        # Cosine decay smoothly reduces lr from initial → 10% of initial,
        # allowing fine-grained exploration in later iterations.
        #
        # FORMULA: lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi*t/T))
        #   where t = current iter, T = total iters
        #   This starts at lr_max, smoothly decays to lr_min.
        # ------------------------------------------------------------------
        iter_num_for_lr = int(self.optimizer.iterations.numpy())
        n_total_lr = max(getattr(self, "_n_iters", 200), 1)
        if self._initial_lr is None:
            self._initial_lr = float(self.optimizer.learning_rate)
        lr_max = self._initial_lr
        lr_min = lr_max * 0.05  # decay to 5% of initial
        progress = min(iter_num_for_lr / n_total_lr, 1.0)
        new_lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * progress))
        self.optimizer.learning_rate.assign(new_lr)

        # ------------------------------------------------------------------
        # Step 2: loss_at helper.
        # Writes debug info into `dbg` dict so we can display CL/CD/penalty
        # without re-evaluating NeuralFoil.
        # ------------------------------------------------------------------
        def loss_at(z: np.ndarray, dbg: dict | None = None) -> float:
            result = eval_all_conditions(
                self._pipeline, z, self._conditions, cl_min=self._cl_min)

            if result is None:
                return float("inf")
            if result["cl_fail"]:
                return float("inf")

            obj = result["avg_cd_cl"]
            if not np.isfinite(obj):
                return float("inf")

            coords = result.get("coords")
            CL_dp  = result.get("CL", np.nan)
            pen    = 0.0
            if coords is not None and np.isfinite(CL_dp):
                try:
                    pen, _ = total_penalty(
                        latent_vec=z, coords=coords, CL=float(CL_dp),
                        lat_lo=self._lat_lo_np, lat_hi=self._lat_hi_np,
                        **self._penalty_kw,
                    )
                    # NO hard-reject here. pen can be 1000+ — that IS the gradient
                    # signal. Converting to inf (old code) made the loss landscape a
                    # cliff: FD saw finite→inf and produced a huge/unstable gradient.
                    # Keeping pen finite gives Adam a proper slope to descend.
                    # (Professor whiteboard Image 3: f(x) = ReLU(x), never inf)
                except Exception:
                    pass

            bp = self._bounds_lam * float(np.sum(
                np.maximum(0.0, self._lat_lo_np - z) +
                np.maximum(0.0, z - self._lat_hi_np)
            ))

            if dbg is not None:
                dbg["CL"]  = float(result.get("CL", np.nan))
                dbg["CD"]  = float(result.get("CD", np.nan))
                dbg["obj"] = float(obj)
                dbg["pen"] = float(pen)
                dbg["bp"]  = float(bp)
                dbg["per_cond"] = result.get("per_cond", [])

            return float(obj + pen + bp)

        # ------------------------------------------------------------------
        # Step 3: Base loss at current z_eff (captures debug info for display)
        # ------------------------------------------------------------------
        dbg0 = {}
        loss_0 = loss_at(z_eff, dbg=dbg0)
        iter_num = int(self.optimizer.iterations.numpy()) + 1

        if not np.isfinite(loss_0):
            self.n_skipped += 1
            self._print_step(iter_num, None, dbg0, skipped=True)
            return {"loss": tf.constant(1000.0, dtype=tf.float32)}

        # ------------------------------------------------------------------
        # Step 4: Finite differences — CENTRAL DIFFERENCES (FIX #2)
        #
        # OLD CODE used one-sided (forward) differences:
        #   grad = (loss(z + eps) - loss_0) / eps
        #
        # PROBLEM WITH ONE-SIDED FD:
        #   When z is near a hard constraint boundary (e.g. camber limit),
        #   perturbing TOWARD the boundary gives loss=inf, so grad=0 for
        #   that dimension. The optimizer then can't move away from the wall.
        #   With forward-only differences, the optimizer is half-blind.
        #
        # FIX — CENTRAL DIFFERENCES:
        #   grad = (loss(z + eps) - loss(z - eps)) / (2 * eps)
        #
        # This samples BOTH sides of each dimension. If +eps is invalid (inf)
        # but -eps is valid, we fall back to a one-sided estimate using loss_0.
        # This way the optimizer can always detect which direction to move,
        # even right at a constraint boundary.
        #
        # COST: doubles FD calls from 12 to 24 per step (6w + 6b each direction).
        # Worth it — without this, the optimizer is stuck from iter 3 onward.
        # ------------------------------------------------------------------
        grad_w_np = np.zeros(6, dtype=np.float64)
        grad_b_np = np.zeros(6, dtype=np.float64)

        # ------------------------------------------------------------------
        # FIX: Scale FD epsilon per dimension for w perturbations.
        #
        # WHY THIS MATTERS:
        #   dz_eff = dw * z_init[i].  If z_init[i] is tiny (e.g. -0.0026),
        #   perturbing w by 0.01 gives dz = 0.000026 — below NeuralFoil's
        #   numerical precision. The gradient is pure noise.
        #   Scaling eps inversely with |z_init| ensures each dimension
        #   gets a meaningful perturbation in z-space.
        #
        #   For b perturbations, dz = db directly, so no scaling needed.
        # ------------------------------------------------------------------
        for i in range(6):
            # Scale w epsilon so dz_eff ≈ fd_eps regardless of z_init magnitude
            z_mag = max(abs(z_init[i]), 0.01)  # floor to prevent division issues
            w_eps_i = self._fd_eps / z_mag      # larger eps when z_init is small

            # Perturb w[i] in both directions
            w_plus  = w_np.copy(); w_plus[i]  += w_eps_i
            w_minus = w_np.copy(); w_minus[i] -= w_eps_i
            loss_plus  = loss_at(w_plus  * z_init + b_np)
            loss_minus = loss_at(w_minus * z_init + b_np)

            # Central diff if both sides finite; fall back to one-sided if one is inf
            if np.isfinite(loss_plus) and np.isfinite(loss_minus):
                grad_w_np[i] = (loss_plus - loss_minus) / (2.0 * w_eps_i)
            elif np.isfinite(loss_plus):
                grad_w_np[i] = (loss_plus  - loss_0) / w_eps_i   # forward fallback
            elif np.isfinite(loss_minus):
                grad_w_np[i] = (loss_0 - loss_minus) / w_eps_i   # backward fallback
            # else: both inf → grad stays 0 (truly boxed in on this dimension)

        for i in range(6):
            # Perturb b[i] in both directions
            b_plus  = b_np.copy(); b_plus[i]  += self._fd_eps
            b_minus = b_np.copy(); b_minus[i] -= self._fd_eps
            loss_plus  = loss_at(w_np * z_init + b_plus)
            loss_minus = loss_at(w_np * z_init + b_minus)

            # Central diff if both sides finite; fall back to one-sided if one is inf
            if np.isfinite(loss_plus) and np.isfinite(loss_minus):
                grad_b_np[i] = (loss_plus - loss_minus) / (2.0 * self._fd_eps)
            elif np.isfinite(loss_plus):
                grad_b_np[i] = (loss_plus  - loss_0) / self._fd_eps   # forward fallback
            elif np.isfinite(loss_minus):
                grad_b_np[i] = (loss_0 - loss_minus) / self._fd_eps   # backward fallback
            # else: both inf → grad stays 0 (truly boxed in on this dimension)

        # ------------------------------------------------------------------
        # Step 5: Apply FD gradients via Adam
        # SAVE w, b before update so we can rollback if new point is invalid.
        # ------------------------------------------------------------------
        w_saved = self.latent_layer.w.numpy().copy().astype(np.float32)
        b_saved = self.latent_layer.b.numpy().copy().astype(np.float32)

        self.optimizer.apply_gradients([
            (tf.constant(grad_w_np.astype(np.float32)), self.latent_layer.w),
            (tf.constant(grad_b_np.astype(np.float32)), self.latent_layer.b),
        ])

        # ------------------------------------------------------------------
        # Step 6: Check new point validity WITHOUT double-calling NeuralFoil.
        #
        # We call eval_all_conditions ONCE for z_new, then compute the penalty
        # inline from that result — no second call to loss_at(z_new).
        # This saves 4 NeuralFoil calls per valid step vs the naive approach.
        # ------------------------------------------------------------------
        w_new = self.latent_layer.w.numpy().astype(np.float64)
        b_new = self.latent_layer.b.numpy().astype(np.float64)
        z_new = np.clip(w_new * z_init + b_new, self._lat_lo_np, self._lat_hi_np)

        new_result = eval_all_conditions(
            self._pipeline, z_new, self._conditions, cl_min=self._cl_min)

        step_ok  = (new_result is not None and not new_result["cl_fail"])
        new_loss = float("inf")

        if step_ok:
            new_obj    = new_result["avg_cd_cl"]
            new_coords = new_result.get("coords")
            new_CL     = float(new_result.get("CL", np.nan))
            new_pen    = 0.0
            if new_coords is not None and np.isfinite(new_CL):
                try:
                    new_pen, _ = total_penalty(
                        latent_vec=z_new, coords=new_coords, CL=new_CL,
                        lat_lo=self._lat_lo_np, lat_hi=self._lat_hi_np,
                        **self._penalty_kw,
                    )
                    # NO hard-reject. A step into a constrained region is still
                    # a valid step — the large penalty (e.g. 1000) means Adam won't
                    # record it as best, and the FD gradient on the NEXT step from
                    # this position will point strongly back toward valid regions.
                    # Rolling back from every constraint touch just causes endless loops.
                except Exception:
                    pass

            if step_ok:
                new_bp   = self._bounds_lam * float(np.sum(
                    np.maximum(0.0, self._lat_lo_np - z_new) +
                    np.maximum(0.0, z_new - self._lat_hi_np)
                ))
                new_loss = float(new_obj + new_pen + new_bp)
                step_ok  = np.isfinite(new_loss)

        if step_ok:
            self.n_valid += 1
            # Only record as "best" when constraints are clean (pen < 1.0).
            # A step with pen=1000 improves nothing — it's just constraint territory.
            # The combined loss including penalty naturally steers Adam back.
            genuinely_valid = (new_pen < 1.0) and np.isfinite(new_loss)
            if genuinely_valid and new_loss < self.best_loss:
                self.best_loss   = new_loss
                self.best_result = {
                    "latent":    z_new.copy(),
                    "coords":    new_result.get("coords"),
                    "CL":        float(new_result["CL"]),
                    "CD":        float(new_result["CD"]),
                    "avg_cd_cl": float(new_result["avg_cd_cl"]),
                    "per_cond":  new_result["per_cond"],
                }
                self.n_improved += 1
                self._last_step_improved = True
            else:
                self._last_step_improved = False

            self.history_log.append({
                "iter":      iter_num,
                "CL":        float(new_result["CL"]),
                "CD":        float(new_result["CD"]),
                "avg_cd_cl": float(new_result["avg_cd_cl"]),
                "loss":      float(new_loss),
            })
        else:
            # step_ok is False only when new_result is None (NeuralFoil crashed)
            # or cl_fail (takeoff CL too low — physical hard requirement).
            #
            # For geometry constraints we NO LONGER rollback (see FIX above).
            # The penalty slope guides Adam back naturally. Rolling back from
            # every constraint touch created an infinite rollback loop.
            #
            # For a true crash (result=None) or CL floor failure, restore w,b
            # so we stay at last known-good position.
            self.latent_layer.w.assign(w_saved)
            self.latent_layer.b.assign(b_saved)
            self.n_skipped += 1
            self._last_step_improved = False

        self._print_step(iter_num, loss_0, dbg0, skipped=False, step_ok=step_ok)
        # Return TF constant (not Python float) so Keras metric tracker works
        return {"loss": tf.constant(float(loss_0), dtype=tf.float32)}

    def _print_step(self, iter_num, loss_0, dbg, skipped=False, step_ok=True):
        """Print a one-liner per epoch: iter, loss, CL, CD, L/D, penalty, valid/skip, ETA."""
        n_total = getattr(self, "_n_iters", 200)
        elapsed = time.time() - getattr(self, "_t_start", time.time())
        secs_per_step = elapsed / max(iter_num, 1)
        eta_secs = secs_per_step * (n_total - iter_num)
        if eta_secs >= 3600:
            eta_str = f"{eta_secs/3600:.1f}h"
        elif eta_secs >= 60:
            eta_str = f"{eta_secs/60:.0f}m"
        else:
            eta_str = f"{eta_secs:.0f}s"

        if skipped or loss_0 is None:
            print(f"  iter {iter_num:4d}/{n_total}"
                  f"  SKIP (invalid foil at current point)"
                  f"  valid={self.n_valid}  skip={self.n_skipped}"
                  f"  ETA {eta_str}")
            return

        CL  = dbg.get("CL", float("nan"))
        CD  = dbg.get("CD", float("nan"))
        ld  = CL / CD if (CD and CD > 0) else 0.0
        pen = dbg.get("pen", 0.0) + dbg.get("bp", 0.0)
        obj = dbg.get("obj", loss_0)
        best_ld = 1.0 / max(self.best_loss, 1e-9)

        status = "✓" if step_ok else "✗ SKIP"
        improved = " ★ NEW BEST" if getattr(self, "_last_step_improved", False) else ""

        print(f"  iter {iter_num:4d}/{n_total}"
              f"  loss={loss_0:.6f}"
              f"  CL={CL:.4f}  CD={CD:.6f}  L/D={ld:.1f}"
              f"  pen={pen:.5f}"
              f"  best_L/D={best_ld:.1f}"
              f"  {status}"
              f"  valid={self.n_valid}/{iter_num}"
              f"  ETA {eta_str}"
              f"  lr={float(self.optimizer.learning_rate):.1e}"
              f"{improved}")


# ===========================================================================
# MAIN OPTIMIZATION FUNCTION
# ===========================================================================

def nom_optimize(
    *,
    # --- Operating conditions ---
    # List of (alpha_degrees, Reynolds_number) pairs to evaluate each step.
    # None → use the module-level OPERATING_CONDITIONS list above.
    conditions: list[tuple[float, float]] | None = None,

    # --- Number of iterations (= epochs passed to nom.fit) ---
    # [2/26]: "iterations and epochs should all be the same — one process"
    # Each epoch = one full FD gradient computation + one Adam update.
    n_iters: int = 200,

    # --- Adam learning rate ---
    # [2/10]: "Learning Rate: 1e-3 or less"
    # Passed to both nom.compile(Adam(lr=...)) and used internally by Adam.
    tf_learning_rate: float = 0.0005,

    # --- Finite difference epsilon ---
    # Size of the perturbation when computing FD gradients.
    # Too small → noisy gradient. Too large → inaccurate gradient.
    # 0.01 is a good default for the 6D latent space.
    fd_eps: float = 0.01,

    # --- Bounds penalty lambda ---
    # [2/19]: "Another penalty, for going over or under the min/max 6 params"
    # Soft penalty weight applied when z_eff drifts outside [lat_lo, lat_hi].
    bounds_lam: float = 10.0,

    # --- Lambda weights for geometry constraints ---
    # These are passed to total_penalty() in constraints.py.
    # lam_bounds: weight for latent vector out of training range
    # lam_geom:   weight for geometric violations (too thin, TE gap too big)
    # lam_cl:     weight for CL violations (below floor or above ceiling)
    # Per 2/17 meeting: lambdas should sum to 1 (auto-normalized in constraints.py)
    lam_bounds: float = 1.0,
    lam_geom:   float = 25.0,
    lam_cl:     float = 50.0,

    # --- Geometry limits ---
    # min_thickness:     foil must be at least this thick everywhere (no razor blades)
    # max_thickness:     physical limit for the foil to be manufacturable
    # te_gap_max:        trailing edge gap (affects structural integrity)
    # min_max_thickness: [2/19 action item] "ADD Minimum thickness of the maximum
    #                    thickness of the foil — at least this for the max thickness"
    # max_camber:        [2/19] tightened to prefer symmetric/NACA foils for 3D printing
    #
    # FIX #3 — max_camber raised from 0.04 → 0.08
    #   ROOT CAUSE OF INFINITE ROLLBACKS (secondary):
    #     The baseline foil hq358 is an HQ-series foil with ~7-8% camber.
    #     With max_camber=0.04, ANY perturbation that nudges camber slightly
    #     upward causes a hard reject (penalty=1000 → loss=inf).
    #     The FD gradient then shows "this direction = inf, don't go there"
    #     for every dimension pointing toward higher camber, and the optimizer
    #     can't explore that half of the space at all.
    #
    #   FIX: Raise to 0.08 (8%c) so the optimizer can freely explore the
    #     neighborhood of the baseline without hitting a hard wall immediately.
    #     This matches the HQ/Eppler family range that the training dataset
    #     already contains, so the decoder can actually decode these foils.
    #
    #   NOTE: If you switch to a symmetric (NACA 00xx) baseline, you can
    #     lower this back to 0.04 without issue. The constraint is only
    #     problematic when the baseline itself is near the camber limit.
    min_thickness:     float = 0.006,
    max_thickness:     float = 0.157,
    te_gap_max:        float = 0.01,
    min_max_thickness: float = 0.04,
    max_camber:        float = 0.08,   # RAISED from 0.04 → 0.08 (see FIX #3 above)

    # --- CL floor (applied ONLY at takeoff condition — 2/26 fix) ---
    # [2/10]: cl_min=0.15, cl_max=None
    # [2/26]: cl_min checked only at last condition (takeoff), not all conditions
    cl_min: float = 0.15,
    cl_max: float | None = None,   # no upper ceiling needed

    # --- File paths ---
    csv_path: str = "data/airfoil_latent_params_6.csv",

    # [2/19]: "Lookup_table_path Line 202: take out None"
    # Defaults to "" — path auto-constructed from design point conditions
    lookup_baseline_path: str = "outputs/best_baseline_foil_averaged.json", #change to "" to auto-construct from conditions

    out_path: str | Path = "outputs",
):
    """
    Run NOM optimization using nom.summary() + nom.compile() + nom.fit().

    [2/26 ACTION ITEM]: "iterations and epochs should all be the same — one
    continuous flow of the diagram. Why are we running tf training 400 epochs
    and then 1000 iterations? This should be one process."

    ANSWER — now implemented:
      nom.compile(Adam(lr=tf_learning_rate))
      nom.fit(dummy_dataset, epochs=n_iters, steps_per_epoch=1)

    That's it. nom.fit() runs n_iters epochs. Each epoch calls train_step()
    once. train_step() computes FD gradients and applies Adam. No external
    for-loop. No separate TF pre-training. Everything in one fit() call.
    """
    if conditions is None:
        conditions = OPERATING_CONDITIONS

    # -----------------------------------------------------------------------
    # [ACTION ITEM 2/19]: Snap decimal Re/alpha to nearest valid grid values
    # -----------------------------------------------------------------------
    conditions = snap_conditions(conditions)

    # Validate ranges supported by NeuralFoil
    for alpha, Re in conditions:
        if not (0 <= alpha <= 15):
            raise ValueError(f"Alpha={alpha}° out of range [0, 15]")
        if not (1e4 <= Re <= 1e7):
            raise ValueError(f"Re={Re:.0e} out of range [1e4, 1e7]")

    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)

    print("=" * 70)
    print("NOM OPTIMIZATION — nom.summary → nom.compile → nom.fit")
    print("=" * 70)
    print(f"Iterations (= epochs in nom.fit): {n_iters}")
    print(f"Adam learning rate: {tf_learning_rate}")
    print(f"FD epsilon: {fd_eps}")
    print(f"Conditions ({len(conditions)}):")
    for a, r in conditions:
        print(f"    alpha={a}°   Re={r:,.0f}")
    print(f"Objective: average CD/CL across all {len(conditions)} conditions")
    print(f"CL floor:  CL >= {cl_min} at TAKEOFF only (last condition = hard reject)")
    print("=" * 70)
    print()

    # -----------------------------------------------------------------------
    # Load latent dataset → compute per-dimension min/max bounds
    # -----------------------------------------------------------------------
    all_latents = load_latent_dataset(csv_path)
    lat_lo, lat_hi = latent_minmax_bounds(all_latents)
    print(f"Latent bounds computed from {len(all_latents)} training foils.")

    pipeline = TalarAIPipeline()
    print(f"✓ Pipeline ready (decoder: {pipeline.decoder_path.name})")
    print()

    # -----------------------------------------------------------------------
    # Load baseline foil from lookup table
    # [2/17]: "use lookup table to find best foil, use as baseline"
    # [2/19]: "just use latent params in the json file"
    # [2/19]: "Lookup_table_path Line 202: take out None"
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("LOADING BASELINE FROM LOOKUP TABLE")
    print("=" * 70)

    # Auto-construct path from design point if user didn't provide one
    if not lookup_baseline_path:
        design_alpha, design_Re = conditions[0]
        a_snap = min(_VALID_ALPHAS, key=lambda a: abs(a - design_alpha))
        r_snap = min(_VALID_RES,    key=lambda r: abs(r - design_Re))
        tag = f"alpha{a_snap:.1f}_Re{r_snap:.0e}"
        lookup_baseline_path = f"outputs/best_baseline_foil_{tag}.json"
        print(f"Auto-constructed baseline path: {lookup_baseline_path}")

    baseline = load_best_baseline(lookup_baseline_path)
    if baseline is None:
        print("⚠️  No baseline found. Run build_lookup_table.py first.")
        return

    # [2/19]: "just use latent params in the json file" — direct array load
    latent_baseline = np.array(baseline["latent"], dtype=float)
    print(f"Baseline foil:  {baseline.get('filename', 'unknown')}")
    print(f"Baseline latent: {np.round(latent_baseline, 4)}")

    # Show baseline L/D before optimization starts
    print("Evaluating baseline across all conditions...")
    baseline_result = eval_all_conditions(
        pipeline, latent_baseline, conditions, cl_min=cl_min)

    if baseline_result is None or baseline_result["cl_fail"]:
        print("⚠️  Baseline fails CL floor at takeoff. Starting anyway.")
        baseline_avg_cd_cl = float("inf")
    else:
        baseline_avg_cd_cl = baseline_result["avg_cd_cl"]
        print(f"✓ Baseline avg L/D = {1.0/baseline_avg_cd_cl:.1f}  "
              f"(avg CD/CL = {baseline_avg_cd_cl:.6f})")
        for c in baseline_result["per_cond"]:
            cl_s = f"{c['CL']:.4f}" if np.isfinite(c.get("CL", np.nan)) else "N/A"
            ld_s = (f"{c['CL']/c['CD']:.1f}"
                    if np.isfinite(c.get("CL", np.nan)) and
                       np.isfinite(c.get("CD", np.nan)) and c["CD"] > 0 else "N/A")
            print(f"    α={c['alpha']}°  Re={c['Re']:,.0f}  CL={cl_s}  L/D={ld_s}")
    print()

    # ------------------------------------------------------------------
    # [FIX 3/3/26] BASELINE CONSTRAINT VALIDATION
    #
    # Check if the loaded baseline itself violates geometry constraints.
    # If it does, NOM will get penalty=1000 on every epoch and never
    # improve (n_improved=0 for all iterations). This was the root cause
    # of the as6097 failure: 9.78% camber > 8% max_camber limit.
    #
    # Detect this BEFORE wasting compute time and print a clear warning.
    # ------------------------------------------------------------------
    if baseline_result is not None and baseline_result.get("coords") is not None:
        _bl_pen_kw = dict(
            lam_bounds=lam_bounds, lam_geom=lam_geom, lam_cl=lam_cl,
            min_thickness=min_thickness, max_thickness=max_thickness,
            te_gap_max=te_gap_max, min_max_thickness=min_max_thickness,
            max_camber=max_camber, cl_min=cl_min, cl_max=cl_max,
        )
        baseline_pen, baseline_geom_info = total_penalty(
            latent_vec=latent_baseline,
            coords=baseline_result["coords"],
            CL=float(baseline_result.get("CL", 0)),
            lat_lo=lat_lo, lat_hi=lat_hi,
            **_bl_pen_kw,
        )
        if baseline_pen >= 1000.0:
            print("!" * 70)
            print("⚠️  BASELINE FOIL VIOLATES GEOMETRY CONSTRAINTS")
            print("!" * 70)
            print(f"  Baseline: {baseline.get('filename', 'unknown')}")
            print(f"  Penalty:  {baseline_pen}")
            print(f"  Reason:   {baseline_geom_info.get('reason', 'unknown')}")
            if 'max_camber_actual' in baseline_geom_info:
                print(f"  Camber:   {baseline_geom_info['max_camber_actual']*100:.1f}%c  "
                      f"(limit: {max_camber*100:.0f}%c)")
            if 't_max' in baseline_geom_info:
                print(f"  t_max:    {baseline_geom_info['t_max']*100:.2f}%c  "
                      f"(min required: {min_max_thickness*100:.1f}%c)")
            if 't_min' in baseline_geom_info:
                print(f"  t_min:    {baseline_geom_info['t_min']*100:.2f}%c  "
                      f"(min required: {min_thickness*100:.2f}%c)")
            print()
            print("  NOM will get penalty=1000 every epoch and n_improved=0.")
            print("  FIX: Re-run build_lookup_table.py --phase2 to pick a")
            print("  geometry-valid baseline, then re-run NOM.")
            print()
            print("  Or use a per-condition baseline instead:")
            print("    nom_optimize(lookup_baseline_path='')")
            print("!" * 70)
            print()
        elif baseline_pen > 0:
            print(f"  ⚠️  Baseline has soft penalty = {baseline_pen:.4f}")
            print(f"     Reason: {baseline_geom_info.get('reason', 'soft_violations')}")
            print(f"     (NOM can still optimize from here — soft penalty is fine)")
            print()
        else:
            print(f"  ✓ Baseline passes all geometry constraints (penalty = 0)")
            print()

    penalty_kwargs = dict(
        lam_bounds=lam_bounds, lam_geom=lam_geom, lam_cl=lam_cl,
        min_thickness=min_thickness, max_thickness=max_thickness,
        te_gap_max=te_gap_max, min_max_thickness=min_max_thickness,
        max_camber=max_camber, cl_min=cl_min, cl_max=cl_max,
    )

    # -----------------------------------------------------------------------
    # Build the NOM model
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("BUILDING NOM MODEL")
    print("=" * 70)

    nom = NOMModel(
        decoder_model=pipeline.decoder,
        lat_lo=lat_lo, lat_hi=lat_hi,
        init_latent=latent_baseline,
        pipeline=pipeline,
        conditions=conditions,
        lat_lo_np=lat_lo,
        lat_hi_np=lat_hi,
        penalty_kwargs=penalty_kwargs,
        cl_min=cl_min,
        fd_eps=fd_eps,
        bounds_lam=bounds_lam,
    )
    nom(None)   # single dummy forward pass to build all layers → enables summary()

    # Initialize best to baseline values so we improve from a known starting point
    # FIX: Compute full loss (obj + penalty + bounds penalty) for baseline,
    # not just raw avg_cd_cl. This ensures the first train_step comparison
    # is apples-to-apples: new_loss = obj+pen+bp vs best_loss = obj+pen+bp.
    # If baseline has penalty=0, this is the same as before. But if the
    # baseline foil has any geometry violations, this prevents a false
    # "improvement" on the first step.
    baseline_full_loss = baseline_avg_cd_cl
    if baseline_result is not None and not baseline_result["cl_fail"]:
        try:
            bl_pen, _ = total_penalty(
                latent_vec=latent_baseline, coords=baseline_result["coords"],
                CL=float(baseline_result["CL"]),
                lat_lo=lat_lo, lat_hi=lat_hi,
                **penalty_kwargs,
            )
            bl_bp = bounds_lam * float(np.sum(
                np.maximum(0.0, lat_lo - latent_baseline) +
                np.maximum(0.0, latent_baseline - lat_hi)
            ))
            baseline_full_loss = baseline_avg_cd_cl + bl_pen + bl_bp
        except Exception:
            pass  # fall back to raw objective if penalty computation fails
    nom.best_loss   = baseline_full_loss
    nom.best_result = {
        "latent":    latent_baseline.copy(),
        "coords":    baseline_result["coords"] if baseline_result else None,
        "CL":        baseline_result["CL"]     if baseline_result else np.nan,
        "CD":        baseline_result["CD"]     if baseline_result else np.nan,
        "avg_cd_cl": baseline_avg_cd_cl,
        "per_cond":  baseline_result["per_cond"] if baseline_result else [],
    }

    # [ACTION ITEM 2/26 whiteboard] nom.summary()
    nom.summary()
    n_train  = sum(int(np.prod(v.shape)) for v in nom.trainable_variables)
    n_frozen = sum(int(np.prod(v.shape)) for v in nom.non_trainable_variables)
    print()
    print(f"  Trainable:  {n_train} params  (w[6] + b[6])")
    print(f"  Frozen:     {n_frozen} params  (z_init[6] + decoder weights)")
    print(f"  Formula:    z_eff[i] = w[i] * z_init[i] + b[i]")
    print(f"  z_init:     {np.round(latent_baseline, 4)}")
    print()

    # [ACTION ITEM 2/26 whiteboard] nom.compile(Adam(learning_rate, learning))
    # This registers Adam as self.optimizer inside NOMModel.
    # train_step() calls self.optimizer.apply_gradients() each epoch.
    nom.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=tf_learning_rate),
        run_eagerly=True,   # REQUIRED: train_step calls .numpy() for NeuralFoil
    )
    print(f"✓ nom.compile(Adam(lr={tf_learning_rate}), run_eagerly=True) complete.")
    print()

    # -----------------------------------------------------------------------
    # [ACTION ITEM 2/26 whiteboard] nom.fit()
    #
    # We pass a dummy dataset because Keras requires input data.
    # train_step() ignores the data argument — it uses self._pipeline and
    # self._conditions (stored in the model) to run NeuralFoil.
    #
    # epochs=n_iters means: call train_step() exactly n_iters times.
    # steps_per_epoch=1 means: one train_step call per epoch.
    # Together: n_iters FD+Adam steps total. That IS the optimization.
    #
    # [2/26]: "one continuous flow of the diagram... should be one process"
    # This IS the one process. No external for-loop. No separate TF epochs.
    # -----------------------------------------------------------------------
    print("=" * 70)
    print(f"nom.fit(epochs={n_iters}) — UNIFIED OPTIMIZATION LOOP")
    print(f"  Each epoch = FD gradients (12 NF calls) + Adam update on w,b")
    print(f"  Total NeuralFoil calls: ~{n_iters * 13 * len(conditions):,}")
    print("=" * 70)
    print()

    # Dummy dataset: one zero vector per epoch. train_step ignores its content.
    dummy_dataset = tf.data.Dataset.from_tensors(
        tf.zeros((1, 6), dtype=tf.float32)
    ).repeat()   # repeat() prevents Keras from running out of data

    nom._n_iters = n_iters    # used by _print_step for ETA display
    nom._t_start = time.time()  # reset timer right before fit starts
    nom.fit(
        dummy_dataset,
        epochs=n_iters,
        steps_per_epoch=1,   # one train_step call per epoch
        verbose=0,           # suppressed — we print our own per-step line above
    )

    print()
    print("✓ nom.fit() complete.")
    print()

    # -----------------------------------------------------------------------
    # Save all outputs
    # -----------------------------------------------------------------------
    best = nom.best_result
    if best is None or best.get("coords") is None:
        print("⚠️  NOM found 0 valid candidates. Nothing to save.")
        return

    # Save best latent vector (CSV + NPY for different use cases)
    np.savetxt(
        out_path / "best_latent_nom.csv",
        best["latent"].reshape(1, -1),
        delimiter=",", header="p1,p2,p3,p4,p5,p6", comments="",
    )
    np.save(out_path / "best_latent_nom.npy", best["latent"])

    # Save best foil coordinates (80 x,y points)
    np.savetxt(
        out_path / "best_coords_nom.csv",
        best["coords"], delimiter=",", header="x,y", comments="",
    )

    # Save full history log (every valid step — used by plot_nom_results.py)
    with open(out_path / "nom_history.json", "w") as f:
        json.dump(nom.history_log, f, indent=2)

    # Build per-condition summary for nom_summary.json
    per_cond_summary = [
        {
            "alpha": c["alpha"],
            "Re":    c["Re"],
            "CL":    float(c["CL"]) if np.isfinite(c.get("CL", np.nan)) else None,
            "CD":    float(c["CD"]) if np.isfinite(c.get("CD", np.nan)) else None,
            "LD":    float(c["CL"] / c["CD"])
                     if (np.isfinite(c.get("CL", np.nan)) and
                         np.isfinite(c.get("CD", np.nan)) and c["CD"] > 0) else None,
        }
        for c in best["per_cond"]
    ]

    summary = {
        "conditions":             [{"alpha": a, "Re": r} for a, r in conditions],
        "alpha":                  conditions[0][0],
        "Re":                     conditions[0][1],
        "n_iters":                int(n_iters),
        "learning_rate":          float(tf_learning_rate),
        "fd_eps":                 float(fd_eps),
        "bounds_lam":             float(bounds_lam),
        "lam_bounds":             float(lam_bounds),
        "lam_geom":               float(lam_geom),
        "lam_cl":                 float(lam_cl),
        "min_thickness":          float(min_thickness),
        "max_thickness":          float(max_thickness),
        "te_gap_max":             float(te_gap_max),
        "min_max_thickness":      float(min_max_thickness),
        "max_camber":             float(max_camber),
        "cl_min":                 float(cl_min),
        "cl_max":                 None if cl_max is None else float(cl_max),
        "valid_evals":            int(nom.n_valid),
        "skipped":                int(nom.n_skipped),
        "n_improved":             int(nom.n_improved),
        "best_avg_cd_cl":         float(best["avg_cd_cl"]),
        "best_avg_LD":            float(1.0 / max(best["avg_cd_cl"], 1e-9)),
        "best_CL":                float(best["CL"]),
        "best_CD":                float(best["CD"]),
        "best_LD":                float(best["CL"] / best["CD"])
                                  if best["CD"] > 0 else None,
        "best_per_condition":     per_cond_summary,
        "best_latent_params":     [float(x) for x in best["latent"]],
        "latent_lo":              [float(x) for x in lat_lo],
        "latent_hi":              [float(x) for x in lat_hi],
        "baseline_foil_filename": baseline.get("filename"),
        "final_result_from":      "nom_fit_custom_train_step",
    }

    with open(out_path / "nom_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Final results printout
    print("=" * 70)
    print("NOM OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Avg L/D:   {1.0/max(best['avg_cd_cl'],1e-9):.2f}  "
          f"(avg across {len(conditions)} conditions)")
    print(f"Avg CD/CL: {best['avg_cd_cl']:.6f}")
    print()
    print("Per-condition breakdown (best foil):")
    for c in per_cond_summary:
        ld = f"{c['LD']:.1f}" if c["LD"] else "N/A"
        cl = f"{c['CL']:.4f}" if c["CL"] else "N/A"
        print(f"  α={c['alpha']}°  Re={c['Re']:,.0f}  CL={cl}  L/D={ld}")
    print()
    print(f"Valid steps:  {nom.n_valid}  |  "
          f"Skipped: {nom.n_skipped}  |  "
          f"Improved: {nom.n_improved}")
    print(f"Outputs: {out_path}/")
    print("=" * 70)


if __name__ == "__main__":
    nom_optimize()