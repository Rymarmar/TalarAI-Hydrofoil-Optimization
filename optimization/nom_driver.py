"""
optimization/nom_driver.py

---------------------------------------------------------------------------
WHAT THIS FILE DOES:
---------------------------------------------------------------------------
The main NOM (Neural Optimization Machine) optimization loop.

Each iteration:
  1. PROPOSE 6 latent parameters [p1..p6]
     - "global": random sample anywhere in the valid dataset range
     - "local":  nudge the current best by a small random step

  2. EVALUATE through the pipeline:
       latent -> decoder -> foil coords -> NeuralFoil -> CL, CD

  3. SCORE:
       objective = CD / CL           (minimize = maximize lift efficiency)
       penalty   = total_penalty()   (geometry, latent bounds, CL window)
       total     = objective + penalty

     KEY: If penalty is float("inf") -- a HARD geometry rejection
     (surfaces crossing, foil too thick/thin, bad camber) -- candidate
     is SKIPPED and can NEVER become the "best" result.

  4. KEEP if total < current best total.

HOW TO RUN:
  python -m optimization.nom_driver

---------------------------------------------------------------------------
WHY THE PREVIOUS RUN PRODUCED A BAD FOIL:
---------------------------------------------------------------------------
Two compounding problems:

  PROBLEM 1 (constraints.py): Geometry violations were SOFT penalties.
    The optimizer could accept a crossing foil if CD/CL was low enough to
    overcome the penalty. Every "NEW BEST" in the old run had tmin < 0
    (surfaces crossed), meaning all 800 shapes were physically invalid.
    FIX: Hard rejects in geometry_penalty() -- inf propagated to driver.

  PROBLEM 2 (nom_driver.py): The inf check was missing.
    Even with inf from constraints, the driver wasn't explicitly skipping
    those candidates before the "update best" step.
    FIX: Explicit `if not np.isfinite(pen): skip` before updating best.

  PROBLEM 3 (parameters): The search was too narrow too fast.
    The old run started local refinement from iteration 30. Since that
    first "best" was already a bad crossing shape, all 800 local steps
    explored the same invalid region of latent space.
    FIX: Smart initialization from actual training dataset foils FIRST,
    plus more global exploration (p_local=0.5 until a valid shape found).
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

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
# Load dataset
# ===========================================================================

def load_latent_dataset(csv_path: str = "data/airfoil_latent_params.csv") -> np.ndarray:
    """
    Load the CSV with 6 latent parameters for every training foil.
    Used to: (1) compute per-parameter min/max bounds, and
             (2) try known-good training foils as starting candidates.

    Returns numpy array (N, 6).
    """
    df = pd.read_csv(csv_path)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] != 6:
        raise ValueError(
            f"Expected 6 numeric columns (p1..p6), got {numeric.shape[1]}.\n"
            f"Columns found: {list(df.columns)}"
        )
    return numeric.values.astype(float)


# ===========================================================================
# Safe pipeline evaluation
# ===========================================================================

def safe_eval(pipeline: TalarAIPipeline,
              latent_vec: np.ndarray,
              *,
              alpha: float,
              Re: float):
    """
    Try to evaluate a candidate latent vector through the full pipeline.
    Returns None if anything produces an invalid result.

    None means "skip" -- this candidate is not scored at all.
    """
    try:
        out = pipeline.eval_latent_with_neuralfoil(latent_vec, alpha=alpha, Re=Re)

        CL     = float(out["CL"])
        CD     = float(out["CD"])
        coords = np.asarray(out["coords"], dtype=float)

        if not (np.isfinite(CL) and np.isfinite(CD)):
            return None
        if CD <= 0.0 or CD > 10.0:
            return None
        if coords.ndim != 2 or coords.shape[1] != 2:
            return None
        if not np.all(np.isfinite(coords)):
            return None

        return {"CL": CL, "CD": CD, "coords": coords}

    except Exception:
        return None


# ===========================================================================
# Candidate proposal strategies
# ===========================================================================

def propose_global(lat_lo: np.ndarray, lat_hi: np.ndarray) -> np.ndarray:
    """
    GLOBAL exploration: random point uniformly in the valid bounds box.
    Used for broad search and escaping local minima.
    """
    return np.random.uniform(lat_lo, lat_hi).astype(float)


def propose_local(best_latent: np.ndarray,
                  *,
                  lr: float,
                  lat_lo: np.ndarray,
                  lat_hi: np.ndarray) -> np.ndarray:
    """
    LOCAL refinement: nudge the current best by a random step of size lr.
    FORMULA: z_new = best + lr * N(0, 1), clipped to [lat_lo, lat_hi]
    Learning rate decays so local steps zoom in as the run progresses.
    """
    step = np.random.normal(0.0, 1.0, size=best_latent.shape).astype(float)
    z    = np.asarray(best_latent, dtype=float) + float(lr) * step
    return np.clip(z, lat_lo, lat_hi).astype(float)


# ===========================================================================
# Smart initialization: try a random subset of training foils first
# ===========================================================================

def find_valid_seed(pipeline: TalarAIPipeline,
                    all_latents: np.ndarray,
                    *,
                    alpha: float,
                    Re: float,
                    lat_lo: np.ndarray,
                    lat_hi: np.ndarray,
                    penalty_kwargs: dict,
                    n_seed_tries: int = 100) -> dict | None:
    """
    Try random foils from the training dataset to find a valid starting point.

    WHY THIS MATTERS:
      In the previous run, the optimizer's first "best" was already a crossing
      foil (tmin < 0). All subsequent local refinement explored that same
      invalid region. This function guarantees we start local refinement from
      a shape that actually passes all hard geometry checks.

    STRATEGY:
      Randomly sample n_seed_tries latent vectors from the dataset
      (not random in the latent space -- these are real training foils).
      Return the one with the lowest total score that passes all hard checks.
    """
    n = len(all_latents)
    indices = np.random.choice(n, size=min(n_seed_tries, n), replace=False)

    best_seed = None

    for idx in indices:
        cand = all_latents[idx].copy()
        res  = safe_eval(pipeline, cand, alpha=alpha, Re=Re)
        if res is None:
            continue

        CL, CD, coords = res["CL"], res["CD"], res["coords"]
        obj = default_objective(CL, CD)
        pen, pen_info = total_penalty(
            latent_vec=cand, coords=coords, CL=CL,
            lat_lo=lat_lo, lat_hi=lat_hi, **penalty_kwargs
        )

        # Skip hard rejects and non-finite objectives
        if not (np.isfinite(pen) and np.isfinite(obj)):
            continue

        total = float(obj + pen)
        if best_seed is None or total < best_seed["total"]:
            best_seed = {
                "iter": 0, "mode": "seed",
                "lr": 0.0,
                "CL": float(CL), "CD": float(CD),
                "objective": float(obj),
                "penalty":   float(pen),
                "total":     total,
                "tmin_int":           float(pen_info.get("min_thickness_int", 0.0)),
                "tmax_int":           float(pen_info.get("max_thickness_int", 0.0)),
                "camber_max_abs_int": float(pen_info.get("max_abs_camber_int", 0.0)),
                "te_gap":             float(pen_info.get("te_gap", 0.0)),
                "le_gap":             float(pen_info.get("le_gap", 0.0)),
                "latent": cand.copy(),
                "coords": coords.copy(),
            }

    return best_seed


# ===========================================================================
# Main NOM optimization loop
# ===========================================================================

def nom_optimize(
    *,
    alpha: float = 6.0,    # Angle of attack in degrees
    Re:    float = 5e5,    # Reynolds number

    n_iters: int = 3000,   # Increased from 800 -- need more global exploration
                            # to find valid shapes in the decoder's output space

    learning_rate_init: float = 1e-3,
    lr_decay:           float = 0.999,

    # p_local: fraction of iterations using local (vs global) proposal.
    # IMPORTANT: During "pre-valid" phase (before any valid best is found),
    # we do 100% global exploration. Once a valid best exists, we switch to
    # p_local/p_global mix. This prevents us from getting stuck near a bad
    # initial point like the old run did.
    p_local: float = 0.75,

    # Lambda weights
    lam_bounds: float = 1.0,
    lam_geom:   float = 25.0,
    lam_cl:     float = 10.0,

    # Geometry thresholds
    # NOTE: max_thickness raised to 0.25 from 0.20 to allow the optimizer
    # to find valid shapes first, then refine. The decoder can produce foils
    # with max_t up to ~0.20 for typical training foils.
    min_thickness:  float = 0.01,
    max_thickness:  float = 0.14,   # tried 0.25
    camber_max_abs: float = 0.12,   # Relaxed from 0.08 -- some foils have
                                     # higher camber and are still physically valid
    te_gap_max:     float = 0.01,
    le_gap_max:     float = 0.01,

    # CL window: can set to None/None to first find ANY valid shape,
    # then tighten in a second run once you know what CL values are reachable.
    cl_min: float | None = 0.30,
    cl_max: float | None = 0.70,

    # Seed search: try training dataset foils before random search
    n_seed_tries: int = 200,    # try up to 200 actual training foils first

    out_dir: str = "outputs",
):
    """
    Run the NOM loop and save the best foil found.

    CHANGES FROM PREVIOUS VERSION:
    --------------------------------
    1. Hard geometry rejection in constraints.py -- invalid shapes are skipped
    2. Explicit inf check in this loop before updating best
    3. Smart seed initialization from training dataset foils
    4. n_iters increased to 3000 for more global exploration
    5. max_thickness relaxed to 0.25, camber_max_abs to 0.12
       (these can be tightened once we confirm valid shapes are found)
    6. 100% global exploration until first valid best is found

    If this still finds 0 valid shapes, try:
      - cl_min=None, cl_max=None  (remove CL constraint entirely)
      - camber_max_abs=0.20
      - max_thickness=0.30
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load dataset and compute bounds
    latents = load_latent_dataset()
    lat_lo, lat_hi = latent_minmax_bounds(latents)

    print("\n=== DATASET LATENT PARAMETER BOUNDS (min/max across ALL foils) ===")
    for i in range(6):
        print(f"  p{i+1}: min = {lat_lo[i]: .6f}, max = {lat_hi[i]: .6f}, "
              f"range = {lat_hi[i]-lat_lo[i]: .6f}")
    print("===================================================================\n")

    pipeline = TalarAIPipeline()

    # Package constraint kwargs for reuse in seed search and main loop
    penalty_kwargs = dict(
        lam_bounds=lam_bounds, lam_geom=lam_geom, lam_cl=lam_cl,
        min_thickness=min_thickness, max_thickness=max_thickness,
        camber_max_abs=camber_max_abs, te_gap_max=te_gap_max,
        le_gap_max=le_gap_max, cl_min=cl_min, cl_max=cl_max,
    )

    # --- Seed search: try actual training foils first ---
    # This guarantees our first "best" is a physically valid foil, so
    # subsequent local refinement explores a good region of latent space.
    print(f"Trying {n_seed_tries} training dataset foils as starting candidates...")
    best = find_valid_seed(pipeline, latents,
                           alpha=alpha, Re=Re,
                           lat_lo=lat_lo, lat_hi=lat_hi,
                           penalty_kwargs=penalty_kwargs,
                           n_seed_tries=n_seed_tries)

    if best is not None:
        print(f"Seed found! total={best['total']:.5f} "
              f"(obj={best['objective']:.5f}, pen={best['penalty']:.5f}) | "
              f"CL={best['CL']:.4f} CD={best['CD']:.6f} | "
              f"tmin={best['tmin_int']:.4f} tmax={best['tmax_int']:.4f} "
              f"camber={best['camber_max_abs_int']:.4f}")
    else:
        print("WARNING: No valid seed found from training dataset.")
        print("         Starting with pure global search.")
        print("         If this fails, loosen min_thickness, max_thickness,")
        print("         or camber_max_abs, or set cl_min=None, cl_max=None.")

    history = []
    valid   = 0
    skipped = 0
    lr      = float(learning_rate_init)

    for it in range(1, n_iters + 1):

        # --- Propose candidate ---
        # If no valid best yet: always go global (explore the whole space)
        # If valid best exists: p_local fraction of the time go local
        use_local = (best is not None) and (np.random.rand() < float(p_local))
        if use_local:
            cand = propose_local(best["latent"], lr=lr, lat_lo=lat_lo, lat_hi=lat_hi)
            mode = "local"
        else:
            cand = propose_global(lat_lo, lat_hi)
            mode = "global"

        # --- Evaluate ---
        res = safe_eval(pipeline, cand, alpha=alpha, Re=Re)
        if res is None:
            skipped += 1
            lr *= float(lr_decay)
            continue

        CL, CD, coords = res["CL"], res["CD"], res["coords"]

        # --- Objective ---
        obj = default_objective(CL, CD)

        # --- Penalty ---
        pen, pen_info = total_penalty(
            latent_vec=cand, coords=coords, CL=CL,
            lat_lo=lat_lo, lat_hi=lat_hi, **penalty_kwargs
        )

        # --- HARD REJECT: skip if penalty is inf or objective is inf ---
        # This is the critical fix. Hard geometry violations (crossing,
        # too thin/thick, bad camber) are completely excluded from
        # consideration. The optimizer cannot trade CD/CL against them.
        if not (np.isfinite(pen) and np.isfinite(obj)):
            skipped += 1
            lr *= float(lr_decay)
            continue

        valid += 1
        total = float(obj + pen)

        rec = {
            "iter":               int(it),
            "mode":               mode,
            "lr":                 float(lr),
            "CL":                 float(CL),
            "CD":                 float(CD),
            "objective":          float(obj),
            "penalty":            float(pen),
            "total":              float(total),
            "tmin_int":           float(pen_info.get("min_thickness_int", 0.0)),
            "tmax_int":           float(pen_info.get("max_thickness_int", 0.0)),
            "camber_max_abs_int": float(pen_info.get("max_abs_camber_int", 0.0)),
            "te_gap":             float(pen_info.get("te_gap", 0.0)),
            "le_gap":             float(pen_info.get("le_gap", 0.0)),
        }
        history.append(rec)

        if best is None or total < best["total"]:
            best = {**rec, "latent": cand.copy(), "coords": coords.copy()}
            print(
                f"[{it:4d}/{n_iters}] NEW BEST | "
                f"total={best['total']:.5f} "
                f"(obj={best['objective']:.5f}, pen={best['penalty']:.5f}) | "
                f"CL={best['CL']:.4f} CD={best['CD']:.6f} | "
                f"tmin={best['tmin_int']:.4f} tmax={best['tmax_int']:.4f} "
                f"camber={best['camber_max_abs_int']:.4f} | "
                f"lr={best['lr']:.2e} mode={best['mode']}"
            )

        lr *= float(lr_decay)

    # --- Save results ---
    if best is None:
        print("\n" + "=" * 60)
        print("NOM found 0 valid candidates after all checks.")
        print("Try loosening these parameters in nom_optimize():")
        print("  cl_min=None, cl_max=None     (remove CL constraint)")
        print("  max_thickness=0.30")
        print("  camber_max_abs=0.20")
        print("  n_iters=5000")
        print("=" * 60)
        return

    np.save(out_path / "best_latent_nom.npy", best["latent"])
    np.savetxt(out_path / "best_latent_nom.csv",
               best["latent"].reshape(1, -1),
               delimiter=",", header="p1,p2,p3,p4,p5,p6", comments="")
    np.savetxt(out_path / "best_coords_nom.csv",
               best["coords"],
               delimiter=",", header="x,y", comments="")

    with open(out_path / "nom_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "alpha": float(alpha), "Re": float(Re),
        "n_iters": int(n_iters),
        "learning_rate_init": float(learning_rate_init),
        "lr_decay": float(lr_decay),
        "p_local": float(p_local),
        "lam_bounds": float(lam_bounds),
        "lam_geom":   float(lam_geom),
        "lam_cl":     float(lam_cl),
        "min_thickness":  float(min_thickness),
        "max_thickness":  float(max_thickness),
        "camber_max_abs": float(camber_max_abs),
        "te_gap_max":     float(te_gap_max),
        "le_gap_max":     float(le_gap_max),
        "cl_min": None if cl_min is None else float(cl_min),
        "cl_max": None if cl_max is None else float(cl_max),
        "n_seed_tries": int(n_seed_tries),
        "valid_evals": int(valid),
        "skipped":     int(skipped),
        "best_total":     float(best["total"]),
        "best_objective": float(best["objective"]),
        "best_penalty":   float(best["penalty"]),
        "best_CL":        float(best["CL"]),
        "best_CD":        float(best["CD"]),
        "best_latent_params": [float(x) for x in best["latent"].reshape(-1)],
        "latent_lo": [float(x) for x in lat_lo],
        "latent_hi": [float(x) for x in lat_hi],
    }

    with open(out_path / "nom_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== NOM Optimization Complete ===")
    print(json.dumps(summary, indent=2))
    print(f"\nAll outputs saved to: {out_path}/")


if __name__ == "__main__":
    nom_optimize()