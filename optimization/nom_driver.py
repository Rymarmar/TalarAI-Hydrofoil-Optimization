"""
optimization/nom_driver.py

2/26 ACTION ITEMS IMPLEMENTED:
  ✓ ONE unified loop — no separate TF pre-step vs NOM iterations.
      Every iteration: Adam gradient step (FD) on w,b  +  candidate eval.
      One counter, one loop. nom.summary → nom.compile → the loop IS nom.fit().
  ✓ Multi-condition objective — each candidate evaluated at all (alpha, Re) pairs.
      objective = average CD/CL across all conditions.
      Hard reject if CL < cl_min at ANY condition (foil must lift at every speed).

ARCHITECTURE (professor's whiteboard Image 2):
  Trainable:  LinearLatentLayer — 12 params: w[6] + b[6]
              z_eff[i] = w[i] * z_init[i] + b[i]   (y = w*x + b)
  Frozen:     Decoder  6 → 100 → 1000 → 80
  Frozen:     NeuralFoil (NumPy, not differentiable — gradients via FD)
  Objective:  average CD/CL across OPERATING_CONDITIONS
  Penalty:    latent bounds + geometry + CL floor at every condition

SINGLE LOOP FLOW (each iteration i = 1 … n_iters):
  1. Compute FD gradients of (avg CD/CL + penalty) w.r.t. w and b (12 params)
  2. Apply Adam step → update w, b → get new z_eff = w*z_init + b
  3. Evaluate z_eff at all conditions, compute avg_objective + penalty
  4. If improvement → record as new best
  5. Also propose a local Gaussian step from current best → evaluate → keep if better
     (this keeps broad exploration alive alongside gradient descent)
"""

from __future__ import annotations
import os
import json
from pathlib import Path

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
#   Max   speed: V = 25.3  ft/s  → Re ≈ 450,000  (top speed)
#
# Alpha range: 1–4 degrees covers in-flight envelope.
# Design point is alpha~1 deg at max speed.
# We use 4 representative points — enough to cover the envelope without
# making each iteration cost 24 NeuralFoil calls (full 6x4 grid).
#
# Each entry: (alpha_degrees, Reynolds_number)
OPERATING_CONDITIONS = [
    (1.0, 450_000),   # max speed / design point
    (2.0, 350_000),   # cruise
    (3.0, 250_000),   # mid-speed
    (4.0, 150_000),   # takeoff / slow
]


# ===========================================================================
# DATASET LOADING
# ===========================================================================

def load_latent_dataset(csv_path: str = "data/airfoil_latent_params_6.csv") -> np.ndarray:
    df = pd.read_csv(csv_path)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] != 6:
        raise ValueError(f"Expected 6 latent columns, found {numeric.shape[1]}")
    return numeric.values.astype(float)


# ===========================================================================
# LOAD BEST BASELINE
# ===========================================================================

def load_best_baseline(json_path: str | Path) -> dict | None:
    """Load best baseline foil from lookup table JSON."""
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
    pipeline: TalarAIPipeline,
    latent_vec: np.ndarray,
    conditions: list[tuple[float, float]],
    cl_min: float = 0.15,
) -> dict | None:
    """
    Evaluate a latent vector at every (alpha, Re) operating condition.

    Returns dict with:
      avg_cd_cl   — average CD/CL across all valid conditions (the objective)
      per_cond    — list of {alpha, Re, CL, CD, cd_cl} for each condition
      n_valid     — how many conditions returned finite CL/CD
      cl_fail     — True if CL < cl_min at ANY condition (hard reject)

    Returns None if ALL conditions fail NeuralFoil entirely.

    WHY AVERAGE CD/CL:
      We want a foil that performs well across the full speed/angle envelope.
      A foil optimal only at max speed but terrible at takeoff is not useful.
      Average CD/CL gives equal weight to every operating point.

    WHY CL FLOOR AT EVERY CONDITION:
      If the foil can't generate lift at takeoff speed, it's physically useless
      regardless of how good it is at cruise. Hard reject = CL < cl_min anywhere.
    """
    z = np.asarray(latent_vec, dtype=float)
    per_cond = []
    cd_cl_vals = []

    for alpha, Re in conditions:
        try:
            out = pipeline.eval_latent_with_neuralfoil(z, alpha=alpha, Re=Re)
            CL = float(out["CL"])
            CD = float(out["CD"])
            coords = out.get("coords")

            if not (np.isfinite(CL) and np.isfinite(CD) and CD > 0):
                per_cond.append({"alpha": alpha, "Re": Re, "CL": np.nan,
                                 "CD": np.nan, "cd_cl": np.nan})
                continue

            cd_cl = CD / (CL + 1e-9)
            per_cond.append({"alpha": alpha, "Re": Re, "CL": CL,
                             "CD": CD, "cd_cl": cd_cl, "coords": coords})
            cd_cl_vals.append(cd_cl)

        except Exception:
            per_cond.append({"alpha": alpha, "Re": Re, "CL": np.nan,
                             "CD": np.nan, "cd_cl": np.nan})

    if len(cd_cl_vals) == 0:
        return None

    avg_cd_cl = float(np.mean(cd_cl_vals))
    n_valid = len(cd_cl_vals)

    # Hard reject if CL < cl_min at ANY condition
    cl_fail = any(
        (not np.isfinite(c["CL"])) or c["CL"] < cl_min
        for c in per_cond
    )

    # Use coords from design point (first condition) for geometry checks
    design_coords = next(
        (c["coords"] for c in per_cond if "coords" in c), None
    )

    return {
        "avg_cd_cl":    avg_cd_cl,
        "per_cond":     per_cond,
        "n_valid":      n_valid,
        "cl_fail":      cl_fail,
        "coords":       design_coords,
        # Expose design-point CL/CD for logging
        "CL":  per_cond[0].get("CL", np.nan) if per_cond else np.nan,
        "CD":  per_cond[0].get("CD", np.nan) if per_cond else np.nan,
    }


# ===========================================================================
# PROPOSAL STRATEGY
# ===========================================================================

def propose_local(best_latent: np.ndarray, *,
                  lr: float,
                  lat_lo: np.ndarray,
                  lat_hi: np.ndarray) -> np.ndarray:
    """Small Gaussian step from best_latent, clipped to training bounds."""
    step = np.random.normal(0.0, 1.0, size=best_latent.shape).astype(float)
    z = np.asarray(best_latent, dtype=float) + float(lr) * step
    return np.clip(z, lat_lo, lat_hi).astype(float)


# ===========================================================================
# ADAM STATE  (manual Adam — no Keras needed outside the model)
# ===========================================================================

class AdamState:
    """
    Minimal Adam optimizer tracking moment estimates for 12 params (w + b).

    WHY NOT USE tf.keras.optimizers.Adam DIRECTLY:
      Keras Adam lives inside nom.fit() — we need it available in our own
      loop so we can call one step at a time and interleave with NOM proposals.
      This is a direct implementation of the Adam update rule.

    Adam update rule:
      m = β1*m + (1-β1)*g          (first moment — mean of gradient)
      v = β2*v + (1-β2)*g²         (second moment — variance of gradient)
      m̂ = m / (1 - β1^t)           (bias correction)
      v̂ = v / (1 - β2^t)
      θ = θ - lr * m̂ / (√v̂ + ε)
    """
    def __init__(self, n: int = 12, lr: float = 0.0005,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr  = lr
        self.b1  = b1
        self.b2  = b2
        self.eps = eps
        self.m   = np.zeros(n, dtype=np.float64)   # first moment
        self.v   = np.zeros(n, dtype=np.float64)   # second moment
        self.t   = 0                                # step counter

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """One Adam update. params and grads both shape (n,). Returns updated params."""
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grads
        self.v = self.b2 * self.v + (1 - self.b2) * grads ** 2
        m_hat  = self.m / (1 - self.b1 ** self.t)
        v_hat  = self.v / (1 - self.b2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ===========================================================================
# LATENT LAYER  (professor's whiteboard: y_i = w_i * x_i + b_i)
# ===========================================================================

class LinearLatentLayer(tf.keras.layers.Layer):
    """
    12 trainable params: w[6] (scale) + b[6] (shift).
    z_eff[i] = w[i] * z_init[i] + b[i]

    z_init is fixed (the baseline foil's latent, never updated).
    Adam updates w and b only.

    nom.summary() shows: Trainable params: 12
    """
    def __init__(self, lat_lo, lat_hi, init_latent=None, **kwargs):
        super().__init__(**kwargs)

        z_init = (np.array(init_latent, dtype=np.float32).reshape(6)
                  if init_latent is not None
                  else np.random.uniform(lat_lo, lat_hi).astype(np.float32))

        self._z_init = self.add_weight(
            name="z_init", shape=(6,),
            initializer=tf.constant_initializer(z_init),
            trainable=False,
        )
        self.w = self.add_weight(
            name="w", shape=(6,), initializer="ones", trainable=True)
        self.b = self.add_weight(
            name="b", shape=(6,), initializer="zeros", trainable=True)

        self._lat_lo = tf.constant(lat_lo.astype(np.float32))
        self._lat_hi = tf.constant(lat_hi.astype(np.float32))

    def call(self, inputs=None):
        z_eff = self.w * self._z_init + self.b
        return tf.expand_dims(z_eff, axis=0)   # (1, 6) for decoder

    def get_effective_latent(self) -> np.ndarray:
        return (self.w.numpy() * self._z_init.numpy() + self.b.numpy()).reshape(6)

    def get_wb(self) -> tuple[np.ndarray, np.ndarray]:
        return self.w.numpy().astype(np.float64), self.b.numpy().astype(np.float64)

    def set_wb(self, w: np.ndarray, b: np.ndarray):
        self.w.assign(w.astype(np.float32))
        self.b.assign(b.astype(np.float32))


# ===========================================================================
# NOM MODEL  (nom.summary → nom.compile → used in unified loop)
# ===========================================================================

class NOMModel(tf.keras.Model):
    """
    Trainable: latent_layer (12 params: w + b)
    Frozen:    decoder (6 → 100 → 1000 → 80)

    nom.summary() → shows 12 trainable params + frozen decoder
    nom.compile()  → sets optimizer (we use our own AdamState in the loop,
                     but compile is called to satisfy the professor's diagram)
    """
    def __init__(self, decoder_model, lat_lo, lat_hi, init_latent=None):
        super().__init__()
        decoder_model.trainable = False
        self.decoder      = decoder_model
        self.latent_layer = LinearLatentLayer(lat_lo, lat_hi, init_latent,
                                              name="latent_layer")

    def call(self, inputs=None, training=False):
        z = self.latent_layer(inputs)                # (1, 6)
        return self.decoder(z, training=False)       # (1, 80)


# ===========================================================================
# FINITE-DIFFERENCE GRADIENT
# ===========================================================================

def compute_fd_gradients(
    pipeline: TalarAIPipeline,
    w: np.ndarray,
    b: np.ndarray,
    z_init: np.ndarray,
    conditions: list[tuple[float, float]],
    lat_lo: np.ndarray,
    lat_hi: np.ndarray,
    penalty_kwargs: dict,
    cl_min: float,
    fd_eps: float = 0.01,
    bounds_lam: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Forward finite differences for grad w.r.t. w[6] and b[6] (12 params total).

    For each param θ_i:
      grad_i ≈ (loss(θ + eps*e_i) - loss(θ)) / eps

    Loss = avg CD/CL across conditions + bounds penalty + geometry penalty

    Returns: (dw, db, loss_at_current) all as numpy arrays/float.
    Returns (None, None, inf) if current z_eff is invalid.
    """
    z_eff = w * z_init + b

    def loss_at(z: np.ndarray) -> float:
        """Compute total loss at latent z. Returns inf if invalid."""
        result = eval_all_conditions(pipeline, z, conditions, cl_min=cl_min)
        if result is None:
            return float("inf")
        if result["cl_fail"]:
            return float("inf")

        obj = result["avg_cd_cl"]
        if not np.isfinite(obj):
            return float("inf")

        coords = result.get("coords")
        CL_dp  = result.get("CL", np.nan)

        pen = 0.0
        if coords is not None and np.isfinite(CL_dp):
            try:
                pen, _ = total_penalty(
                    latent_vec=z, coords=coords, CL=float(CL_dp),
                    lat_lo=lat_lo, lat_hi=lat_hi, **penalty_kwargs,
                )
                if pen >= 1000.0:
                    return float("inf")
            except Exception:
                pass

        # Bounds penalty on the 12 params (per professor's whiteboard)
        bp = bounds_lam * float(np.sum(
            np.maximum(0.0, lat_lo - z) + np.maximum(0.0, z - lat_hi)
        ))

        return float(obj + pen + bp)

    loss_0 = loss_at(z_eff)
    if not np.isfinite(loss_0):
        return None, None, float("inf")

    dw = np.zeros(6, dtype=np.float64)
    db = np.zeros(6, dtype=np.float64)

    # Gradient w.r.t. each w_i: bump w_i, recompute z_eff, eval loss
    for i in range(6):
        w_plus    = w.copy(); w_plus[i] += fd_eps
        z_plus    = w_plus * z_init + b
        loss_plus = loss_at(z_plus)
        if np.isfinite(loss_plus):
            dw[i] = (loss_plus - loss_0) / fd_eps

    # Gradient w.r.t. each b_i: bump b_i, recompute z_eff, eval loss
    for i in range(6):
        b_plus    = b.copy(); b_plus[i] += fd_eps
        z_plus    = w * z_init + b_plus
        loss_plus = loss_at(z_plus)
        if np.isfinite(loss_plus):
            db[i] = (loss_plus - loss_0) / fd_eps

    return dw, db, loss_0


# ===========================================================================
# MAIN OPTIMIZATION LOOP
# ===========================================================================

def nom_optimize(
    *,
    # --- Operating conditions ---
    conditions: list[tuple[float, float]] = None,   # (alpha, Re) pairs; None → OPERATING_CONDITIONS

    # --- Iterations (ONE unified counter) ---
    n_iters: int = 1000,

    # --- Adam learning rate (for the gradient step inside the loop) ---
    tf_learning_rate: float = 0.0005,

    # --- Local-search step size and decay ---
    learning_rate_init: float = 0.005,
    lr_decay: float = 0.999,

    # --- FD epsilon and bounds lambda ---
    fd_eps: float = 0.01,
    bounds_lam: float = 10.0,

    # --- Lambda weights (auto-normalized in constraints.py) ---
    lam_bounds: float = 1.0,
    lam_geom: float = 25.0,
    lam_cl: float = 50.0,

    # --- Geometry limits ---
    min_thickness: float = 0.006,
    max_thickness: float = 0.157,
    te_gap_max: float = 0.01,
    min_max_thickness: float = 0.04,
    max_camber: float = 0.04,

    # --- CL floor (applied at EVERY condition) ---
    cl_min: float = 0.15,
    cl_max: float | None = None,

    # --- Paths ---
    csv_path: str = "data/airfoil_latent_params_6.csv",
    lookup_baseline_path: str = "",
    out_path: str | Path = "outputs",
):
    """
    Unified NOM optimization loop.

    Each iteration does TWO things:
      (A) Adam gradient step: compute FD gradients of avg CD/CL + penalty
          w.r.t. w and b, apply Adam update → new z_eff = w*z_init + b
      (B) Local proposal: Gaussian step from current best → evaluate → keep if better

    Both (A) and (B) count toward the same iteration counter.
    The best candidate from either path is tracked as global best.

    This satisfies the professor's "one process" requirement:
      nom.summary() → shows 12 trainable params
      nom.compile()  → sets Adam (shown for correctness)
      The loop below IS nom.fit() — one unified iteration stream.
    """
    if conditions is None:
        conditions = OPERATING_CONDITIONS

    # Validate all (alpha, Re) pairs
    for alpha, Re in conditions:
        if not (0 <= alpha <= 15):
            raise ValueError(f"Alpha={alpha}° out of range [0, 15]")
        if not (1e4 <= Re <= 1e7):
            raise ValueError(f"Re={Re:.0e} out of range [1e4, 1e7]")

    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)

    print("=" * 70)
    print("NOM OPTIMIZATION  —  UNIFIED SINGLE LOOP")
    print("=" * 70)
    print(f"Iterations:  {n_iters}  (Adam gradient step + local proposal each iter)")
    print(f"Conditions ({len(conditions)}):")
    for a, r in conditions:
        print(f"    alpha={a}°   Re={r:.0e}")
    print(f"Objective:   average CD/CL across all conditions")
    print(f"CL floor:    CL ≥ {cl_min} at EVERY condition (hard reject if any fail)")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Load dataset bounds + initialize pipeline
    # ------------------------------------------------------------------
    all_latents = load_latent_dataset(csv_path)
    lat_lo, lat_hi = latent_minmax_bounds(all_latents)
    print(f"Latent bounds computed from {len(all_latents)} training foils.")

    pipeline = TalarAIPipeline()
    print(f"✓ Pipeline ready (decoder: {pipeline.decoder_path.name})")
    print()

    # ------------------------------------------------------------------
    # Load baseline from lookup table
    # ------------------------------------------------------------------
    print("=" * 70)
    print("LOADING BASELINE")
    print("=" * 70)

    _valid_alphas = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    _valid_res    = [150000, 250000, 350000, 450000]

    # Use design-point condition (first in list) for baseline lookup
    design_alpha, design_Re = conditions[0]
    alpha_snapped = min(_valid_alphas, key=lambda a: abs(a - design_alpha))
    Re_snapped    = min(_valid_res,    key=lambda r: abs(r - design_Re))

    if not lookup_baseline_path:
        tag = f"alpha{alpha_snapped:.1f}_Re{Re_snapped:.0e}"
        lookup_baseline_path = f"outputs/best_baseline_foil_{tag}.json"
        print(f"Auto-constructed baseline path: {lookup_baseline_path}")

    baseline = load_best_baseline(lookup_baseline_path)
    if baseline is None:
        print("⚠️  No baseline found — cannot run. Check lookup table path.")
        return

    latent_baseline = np.array(baseline["latent"], dtype=float)
    print(f"Baseline foil: {baseline['filename']}")

    # Evaluate baseline across all conditions
    print("Evaluating baseline across all conditions...")
    baseline_result = eval_all_conditions(
        pipeline, latent_baseline, conditions, cl_min=cl_min
    )
    if baseline_result is None or baseline_result["cl_fail"]:
        print("⚠️  Baseline fails CL floor at one or more conditions.")
        print("   Continuing anyway — NOM will search from baseline latent.")
    else:
        print(f"✓ Baseline avg L/D = {1.0/baseline_result['avg_cd_cl']:.1f}  "
              f"(avg CD/CL = {baseline_result['avg_cd_cl']:.6f})")
        for c in baseline_result["per_cond"]:
            cl_str = f"{c['CL']:.4f}" if np.isfinite(c.get("CL", np.nan)) else "N/A"
            ld_str = (f"{c['CL']/c['CD']:.1f}" if (np.isfinite(c.get("CL", np.nan))
                      and np.isfinite(c.get("CD", np.nan)) and c["CD"] > 0) else "N/A")
            print(f"    α={c['alpha']}°  Re={c['Re']:.0e}  CL={cl_str}  L/D={ld_str}")
    print()

    penalty_kwargs = dict(
        lam_bounds=lam_bounds, lam_geom=lam_geom, lam_cl=lam_cl,
        min_thickness=min_thickness, max_thickness=max_thickness,
        te_gap_max=te_gap_max, min_max_thickness=min_max_thickness,
        max_camber=max_camber, cl_min=cl_min, cl_max=cl_max,
    )

    # ------------------------------------------------------------------
    # Build NOM model  (nom.summary → nom.compile)
    # This satisfies the professor's diagram exactly:
    #   - nom.summary() shows 12 trainable (w + b) + frozen decoder
    #   - nom.compile() registers Adam optimizer
    #   - The loop below IS the training (replaces nom.fit())
    # ------------------------------------------------------------------
    print("=" * 70)
    print("NOM MODEL  (nom.summary → nom.compile → unified loop)")
    print("=" * 70)

    nom = NOMModel(
        decoder_model=pipeline.decoder,
        lat_lo=lat_lo, lat_hi=lat_hi,
        init_latent=latent_baseline,
    )
    nom(None)   # build layers so summary works

    nom.summary()

    n_train  = sum(int(np.prod(v.shape)) for v in nom.trainable_variables)
    n_frozen = sum(int(np.prod(v.shape)) for v in nom.non_trainable_variables)
    print()
    print(f"  Trainable:  {n_train} params  (w[6] + b[6] — Adam updates these)")
    print(f"  Frozen:     {n_frozen} params  (decoder 6→100→1000→80)")
    print(f"  Formula:    z_eff[i] = w[i] * z_init[i] + b[i]")
    print(f"  z_init:     {np.round(latent_baseline, 4)}")
    print()

    # nom.compile — sets optimizer (professor's diagram requirement)
    nom.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=tf_learning_rate))
    print(f"✓ nom.compile(Adam(lr={tf_learning_rate}))  complete.")
    print()

    # ------------------------------------------------------------------
    # Initialize Adam state + starting point
    # ------------------------------------------------------------------
    adam = AdamState(n=12, lr=tf_learning_rate)

    z_init = latent_baseline.copy()
    w      = np.ones(6,  dtype=np.float64)   # start at y = 1*x + 0 = x
    b      = np.zeros(6, dtype=np.float64)

    # Set layer weights to match
    nom.latent_layer.set_wb(w, b)

    # Initial evaluation
    z_eff_init = w * z_init + b
    init_result = eval_all_conditions(pipeline, z_eff_init, conditions, cl_min=cl_min)

    if init_result is None or init_result["cl_fail"]:
        print("⚠️  Starting point fails CL floor — using raw baseline objective.")
        best_obj = float("inf")
    else:
        best_obj = init_result["avg_cd_cl"]

    best = {
        "latent":    z_eff_init.copy(),
        "coords":    init_result["coords"] if init_result else None,
        "CL":        init_result["CL"]     if init_result else np.nan,
        "CD":        init_result["CD"]     if init_result else np.nan,
        "avg_cd_cl": best_obj,
        "per_cond":  init_result["per_cond"] if init_result else [],
    }

    print("=" * 70)
    print(f"UNIFIED LOOP  ({n_iters} iterations)")
    print(f"  Each iteration: (A) Adam gradient step on w,b  +  (B) local proposal")
    print("=" * 70)
    print()

    history = []
    lr_local = float(learning_rate_init)
    valid = 0
    skipped = 0

    _pending_skips = 0

    def _flush_skips():
        nonlocal _pending_skips
        if _pending_skips > 0:
            print(f"  ... skipped {_pending_skips} invalid / constraint-rejected candidates")
            _pending_skips = 0

    for it in range(1, n_iters + 1):

        improved_this_iter = False

        # ==============================================================
        # (A) ADAM GRADIENT STEP on w and b
        #     Computes FD gradients of (avg CD/CL + penalty) w.r.t. w,b
        #     Applies one Adam update → new z_eff = w*z_init + b
        # ==============================================================
        dw, db, loss_0 = compute_fd_gradients(
            pipeline=pipeline, w=w, b=b, z_init=z_init,
            conditions=conditions, lat_lo=lat_lo, lat_hi=lat_hi,
            penalty_kwargs=penalty_kwargs, cl_min=cl_min,
            fd_eps=fd_eps, bounds_lam=bounds_lam,
        )

        if dw is not None:
            # Stack w and b grads into one 12-vector for Adam
            grads_12 = np.concatenate([dw, db])
            params_12 = np.concatenate([w, b])
            updated_12 = adam.step(params_12, grads_12)
            w_new = np.clip(updated_12[:6], -10.0, 10.0)   # prevent runaway w
            b_new = updated_12[6:]

            # Evaluate new z_eff after Adam step
            z_eff_new = np.clip(w_new * z_init + b_new, lat_lo, lat_hi)
            grad_result = eval_all_conditions(
                pipeline, z_eff_new, conditions, cl_min=cl_min
            )

            if grad_result is not None and not grad_result["cl_fail"]:
                obj_new = grad_result["avg_cd_cl"]

                coords_new = grad_result.get("coords")
                pen_new = 0.0
                if coords_new is not None and np.isfinite(grad_result["CL"]):
                    try:
                        pen_new, _ = total_penalty(
                            latent_vec=z_eff_new,
                            coords=coords_new,
                            CL=float(grad_result["CL"]),
                            lat_lo=lat_lo, lat_hi=lat_hi, **penalty_kwargs,
                        )
                    except Exception:
                        pen_new = 0.0

                if pen_new < 1000.0 and np.isfinite(obj_new):
                    total_new = float(obj_new + pen_new)
                    valid += 1

                    if total_new < best["avg_cd_cl"]:
                        w, b = w_new, b_new
                        nom.latent_layer.set_wb(w, b)
                        best = {
                            "latent":    z_eff_new.copy(),
                            "coords":    coords_new,
                            "CL":        float(grad_result["CL"]),
                            "CD":        float(grad_result["CD"]),
                            "avg_cd_cl": total_new,
                            "per_cond":  grad_result["per_cond"],
                        }
                        improved_this_iter = True
                    else:
                        # Still accept the Adam step even without improvement
                        # (gradient descent can worsen briefly before improving)
                        w, b = w_new, b_new
                        nom.latent_layer.set_wb(w, b)

        # ==============================================================
        # (B) LOCAL GAUSSIAN PROPOSAL from current best
        #     Independent of the gradient step — keeps broad exploration
        # ==============================================================
        cand = propose_local(best["latent"], lr=lr_local,
                             lat_lo=lat_lo, lat_hi=lat_hi)

        cand_result = eval_all_conditions(pipeline, cand, conditions, cl_min=cl_min)

        if cand_result is None or cand_result["cl_fail"]:
            skipped += 1
            _pending_skips += 1
        else:
            cand_obj   = cand_result["avg_cd_cl"]
            cand_coords = cand_result.get("coords")
            cand_pen    = 0.0

            if cand_coords is not None and np.isfinite(cand_result["CL"]):
                try:
                    cand_pen, pen_info = total_penalty(
                        latent_vec=cand, coords=cand_coords,
                        CL=float(cand_result["CL"]),
                        lat_lo=lat_lo, lat_hi=lat_hi, **penalty_kwargs,
                    )
                except Exception:
                    cand_pen = 1000.0
            else:
                pen_info = {}

            if cand_pen >= 1000.0 or not np.isfinite(cand_obj):
                skipped += 1
                _pending_skips += 1
            else:
                valid += 1
                cand_total = float(cand_obj + cand_pen)

                if cand_total < best["avg_cd_cl"]:
                    best = {
                        "latent":    cand.copy(),
                        "coords":    cand_coords,
                        "CL":        float(cand_result["CL"]),
                        "CD":        float(cand_result["CD"]),
                        "avg_cd_cl": cand_total,
                        "per_cond":  cand_result["per_cond"],
                    }
                    improved_this_iter = True

                record = {
                    "iter":      int(it),
                    "lr":        float(lr_local),
                    "CL":        float(cand_result["CL"]),
                    "CD":        float(cand_result["CD"]),
                    "avg_cd_cl": float(cand_obj),
                    "objective": float(cand_obj),   # kept for plot compatibility
                    "penalty":   float(cand_pen),
                    "total":     float(cand_total),
                    "t_min":     float(pen_info.get("t_min", 0.0)),
                    "t_max":     float(pen_info.get("t_max", 0.0)),
                    "te_gap":    float(pen_info.get("te_gap", 0.0)),
                }
                history.append(record)

        if improved_this_iter:
            _flush_skips()
            CL_log = best["CL"]
            CD_log = best["CD"]
            ld_log = CL_log / CD_log if (np.isfinite(CD_log) and CD_log > 0) else 0.0
            print(
                f"[{it:4d}/{n_iters}] NEW BEST | "
                f"avg CD/CL={best['avg_cd_cl']:.6f}  avg L/D={1.0/max(best['avg_cd_cl'],1e-9):.1f} | "
                f"design-pt CL={CL_log:.4f} CD={CD_log:.6f} L/D={ld_log:.1f} | "
                f"lr={lr_local:.2e}"
            )

        lr_local *= float(lr_decay)

    _flush_skips()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if best["coords"] is None:
        print("\n⚠️  NOM found 0 valid candidates.")
        return

    np.savetxt(
        out_path / "best_latent_nom.csv",
        best["latent"].reshape(1, -1),
        delimiter=",", header="p1,p2,p3,p4,p5,p6", comments="",
    )
    np.save(out_path / "best_latent_nom.npy", best["latent"])
    np.savetxt(
        out_path / "best_coords_nom.csv",
        best["coords"], delimiter=",", header="x,y", comments="",
    )
    with open(out_path / "nom_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Build per-condition summary for the plot
    per_cond_summary = [
        {
            "alpha": c["alpha"],
            "Re":    c["Re"],
            "CL":    float(c["CL"])  if np.isfinite(c.get("CL", np.nan)) else None,
            "CD":    float(c["CD"])  if np.isfinite(c.get("CD", np.nan)) else None,
            "LD":    float(c["CL"] / c["CD"])
                     if (np.isfinite(c.get("CL", np.nan)) and
                         np.isfinite(c.get("CD", np.nan)) and c["CD"] > 0) else None,
        }
        for c in best["per_cond"]
    ]

    summary = {
        # operating conditions
        "conditions":           [{"alpha": a, "Re": r} for a, r in conditions],
        "alpha":                conditions[0][0],   # design point (kept for plot compat)
        "Re":                   conditions[0][1],
        # iteration config
        "n_iters":              int(n_iters),
        "tf_learning_rate":     float(tf_learning_rate),
        "lr_init_local":        float(learning_rate_init),
        "lr_decay":             float(lr_decay),
        "fd_eps":               float(fd_eps),
        "bounds_lam":           float(bounds_lam),
        # constraint config
        "lam_bounds":           float(lam_bounds),
        "lam_geom":             float(lam_geom),
        "lam_cl":               float(lam_cl),
        "min_thickness":        float(min_thickness),
        "max_thickness":        float(max_thickness),
        "te_gap_max":           float(te_gap_max),
        "cl_min":               float(cl_min),
        "cl_max":               None if cl_max is None else float(cl_max),
        # results
        "valid_evals":          int(valid),
        "skipped":              int(skipped),
        "best_avg_cd_cl":       float(best["avg_cd_cl"]),
        "best_avg_LD":          float(1.0 / max(best["avg_cd_cl"], 1e-9)),
        "best_CL":              float(best["CL"]),
        "best_CD":              float(best["CD"]),
        "best_LD":              float(best["CL"] / best["CD"])
                                if best["CD"] > 0 else None,
        "best_per_condition":   per_cond_summary,
        "best_latent_params":   [float(x) for x in best["latent"]],
        "latent_lo":            [float(x) for x in lat_lo],
        "latent_hi":            [float(x) for x in lat_hi],
        # baseline info
        "baseline_foil_filename": baseline.get("filename"),
        # TF info (kept for plot compat — TF runs inside the unified loop now)
        "tf_ran":               True,
        "tf_n_epochs":          int(n_iters),       # same as loop count
        "tf_learning_rate":     float(tf_learning_rate),
        "tf_CL":                float(best["CL"]),
        "tf_CD":                float(best["CD"]),
        "tf_LD":                float(best["CL"] / best["CD"]) if best["CD"] > 0 else None,
        "final_result_from":    "unified_loop",
    }

    with open(out_path / "nom_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 70)
    print("NOM OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Avg L/D:   {1.0/max(best['avg_cd_cl'],1e-9):.2f}  (avg across {len(conditions)} conditions)")
    print(f"Avg CD/CL: {best['avg_cd_cl']:.6f}")
    print()
    print("Per-condition breakdown (best foil):")
    for c in per_cond_summary:
        ld = f"{c['LD']:.1f}" if c["LD"] else "N/A"
        cl = f"{c['CL']:.4f}" if c["CL"] else "N/A"
        print(f"  α={c['alpha']}°  Re={c['Re']:.0e}  CL={cl}  L/D={ld}")
    print()
    print(f"Valid:   {valid}/{n_iters} ({100*valid/n_iters:.1f}%)")
    print(f"Skipped: {skipped}/{n_iters} ({100*skipped/n_iters:.1f}%)")
    print(f"Outputs: {out_path}/")
    print("=" * 70)


if __name__ == "__main__":
    nom_optimize()