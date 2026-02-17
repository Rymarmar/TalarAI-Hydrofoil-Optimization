"""
optimization/nom_driver.py

---------------------------------------------------------------------------
WHAT THIS FILE DOES (plain English):
---------------------------------------------------------------------------
This is the main NOM (Neural Optimization Machine) loop. It searches
through the 6-dimensional latent space to find the foil shape that minimizes
CD/CL (drag-to-lift ratio) while satisfying all physical constraints.

EVERY iteration does these steps:
  1. PROPOSE a candidate set of 6 latent parameters [p1..p6]
       - "global" strategy: random point anywhere in the valid dataset range
       - "local"  strategy: small random nudge from the current best result

  2. EVALUATE through the full pipeline:
       latent [p1..p6]
           --> decoder (talarai_pipeline.py)
           --> 80-point foil coordinates
           --> NeuralFoil (physics-informed neural net)
           --> CL (lift coefficient), CD (drag coefficient)

  3. SCORE the candidate:
       objective = CD / CL          (we want to MINIMIZE this = maximize L/D)
       penalty   = total_penalty()  (from constraints.py: geometry + bounds + CL)
       total     = objective + penalty

       CRITICAL: If penalty = inf (hard geometry violation like surfaces crossing,
       foil too thick/thin, or bad camber), the candidate is SKIPPED ENTIRELY.
       The optimizer can never accept a physically impossible foil.

  4. KEEP the candidate if total < current best total. Replace best.

---------------------------------------------------------------------------
HOW TO RUN:
  python -m optimization.nom_driver

---------------------------------------------------------------------------
HOW THE BOUNDS WORK (professor's action item):
---------------------------------------------------------------------------
  The 6 latent parameters have different natural ranges depending on what
  part of the foil shape they control. Rather than guessing a fixed range
  (e.g., always [-1, 1]), we:
    1. Load the full dataset CSV (one row per training airfoil, 6 columns)
    2. Find the actual min and max of EACH of the 6 parameters across all foils
    3. Use those real data min/max as the optimizer's allowed bounds
  This way the optimizer only searches shapes the decoder has actually seen,
  not made-up extrapolations.

---------------------------------------------------------------------------
WHY WE DO A "SEED SEARCH" BEFORE THE MAIN LOOP:
---------------------------------------------------------------------------
  In a previous run, the optimizer's very FIRST "best" was already a bad foil
  with surfaces crossing (tmin < 0). Because 75% of subsequent proposals
  are LOCAL refinements around the current best, all 800 iterations explored
  the same invalid region of latent space and the output was garbage.

  FIX: Before the main loop, try n_seed_tries random foils from the TRAINING
  DATASET (these are known real foils). Use the best valid one as the
  starting point. This guarantees local refinement begins from a good foil.
---------------------------------------------------------------------------
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Import the pipeline (handles decoder + NeuralFoil evaluation)
try:
    from pipeline.talarai_pipeline import TalarAIPipeline
except ModuleNotFoundError:
    from talarai_pipeline import TalarAIPipeline

# Import objective function and constraint penalties
try:
    from optimization.objective import default_objective
    from optimization.constraints import latent_minmax_bounds, total_penalty
except ModuleNotFoundError:
    from objective import default_objective
    from constraints import latent_minmax_bounds, total_penalty


# ===========================================================================
# STEP 0: Load the dataset of latent parameters
# ===========================================================================

def load_latent_dataset(csv_path: str = "data/airfoil_latent_params.csv") -> np.ndarray:
    """
    PURPOSE:
      Load the CSV file that contains the 6 latent parameters (p1..p6) for
      every airfoil in our training dataset.

    WHY WE NEED THIS:
      1. To compute the per-parameter min/max bounds (so the optimizer stays
         inside the region the decoder was trained on).
      2. To try actual known-good foils as starting candidates (seed search),
         which guarantees the first "best" is a physically real foil.

    HOW IT CONNECTS:
      Called once at the start of nom_optimize(). The returned array is passed
      to latent_minmax_bounds() for bounds, and to find_valid_seed() for seeds.

    INPUTS:
      csv_path -- path to the CSV file (one row per foil, 6 numeric columns)

    OUTPUT:
      numpy array of shape (N, 6): N foils x 6 latent parameters
    """
    df = pd.read_csv(csv_path)

    # Select only the numeric columns (p1..p6), skip any filename/label columns
    numeric = df.select_dtypes(include=[np.number])

    if numeric.shape[1] != 6:
        raise ValueError(
            f"Expected exactly 6 numeric columns (p1..p6), found {numeric.shape[1]}.\n"
            f"Columns in file: {list(df.columns)}"
        )

    return numeric.values.astype(float)  # shape: (N, 6)


# ===========================================================================
# STEP 1: Safe pipeline evaluation
# ===========================================================================

def safe_eval(pipeline: TalarAIPipeline,
              latent_vec: np.ndarray,
              *,
              alpha: float,
              Re: float):
    """
    PURPOSE:
      Try to run a candidate latent vector through the full pipeline:
          latent -> decoder -> coords -> NeuralFoil -> CL, CD

      Returns None if ANYTHING goes wrong (decoder crash, non-finite output,
      bad coordinate shape, etc.). None means "skip this candidate."

    WHY WE NEED THIS:
      Some latent vectors -- especially far from the training distribution --
      cause the decoder to produce nonsense (NaN, extreme shapes). NeuralFoil
      can also return non-finite values for these shapes. Rather than crashing
      the whole optimization, we simply skip bad candidates.

    HOW IT CONNECTS:
      Called every iteration in nom_optimize() and in find_valid_seed().
      If it returns None, the iteration is skipped (skipped counter goes up).
      If it returns a dict, we proceed to scoring and constraint checking.

    INPUTS:
      pipeline   -- TalarAIPipeline object (owns the decoder model)
      latent_vec -- shape (6,): candidate latent params to evaluate
      alpha      -- angle of attack in degrees (fixed for the optimization run)
      Re         -- Reynolds number (fixed for the optimization run)

    OUTPUT:
      dict with keys {"CL", "CD", "coords"} if successful, or None if failed.
    """
    try:
        out = pipeline.eval_latent_with_neuralfoil(latent_vec, alpha=alpha, Re=Re)

        CL     = float(out["CL"])
        CD     = float(out["CD"])
        coords = np.asarray(out["coords"], dtype=float)

        # Reject non-finite aerodynamic values
        if not (np.isfinite(CL) and np.isfinite(CD)):
            return None

        # Reject non-physical drag values
        # CD <= 0 means NeuralFoil returned a nonsensical result
        # CD > 10 is wildly outside any real foil's drag range
        if CD <= 0.0 or CD > 10.0:
            return None

        # Reject malformed coordinate arrays
        if coords.ndim != 2 or coords.shape[1] != 2:
            return None

        # Reject coordinates containing NaN or Inf
        if not np.all(np.isfinite(coords)):
            return None

        return {"CL": CL, "CD": CD, "coords": coords}

    except Exception:
        # Catch any crash from the decoder or NeuralFoil
        return None


# ===========================================================================
# STEP 2: Proposal strategies (how we generate the next candidate to try)
# ===========================================================================

def propose_global(lat_lo: np.ndarray, lat_hi: np.ndarray) -> np.ndarray:
    """
    PURPOSE:
      Generate a new candidate latent vector by sampling UNIFORMLY at RANDOM
      within the allowed bounds box [lat_lo, lat_hi].

    WHY:
      Global proposals explore the entire valid design space and help the
      optimizer escape local minima. We use this when we have no good
      starting point yet, or when we want broad exploration.

    HOW IT CONNECTS:
      Called in nom_optimize() when use_local=False (global proposal mode).

    INPUTS:
      lat_lo -- shape (6,): lower bound for each of the 6 latent params
      lat_hi -- shape (6,): upper bound for each of the 6 latent params

    OUTPUT:
      shape (6,): a random latent vector inside the valid bounds box
    """
    return np.random.uniform(lat_lo, lat_hi).astype(float)


def propose_local(best_latent: np.ndarray,
                  *,
                  lr: float,
                  lat_lo: np.ndarray,
                  lat_hi: np.ndarray) -> np.ndarray:
    """
    PURPOSE:
      Generate a new candidate by nudging the CURRENT BEST latent vector by
      a small random step. This is local refinement around the best result.

    FORMULA:
      z_new = best_latent + lr * N(0, 1)   then clipped to [lat_lo, lat_hi]

      N(0, 1) = random number from a standard normal distribution (mean=0, std=1)
      lr      = learning rate: controls the step size. Starts at learning_rate_init
                and decays by lr_decay each iteration, so early steps are large
                (broad local search) and late steps are tiny (fine-tuning).

    WHY:
      Once we have a good foil shape, local refinement makes small improvements
      to it rather than jumping to a completely random new shape. This is more
      efficient for fine-tuning.

    HOW IT CONNECTS:
      Called in nom_optimize() when use_local=True (local proposal mode).
      The step size lr is passed in from the main loop where it decays over time.

    INPUTS:
      best_latent -- shape (6,): the current best latent vector to refine from
      lr          -- learning rate (step size for the random perturbation)
      lat_lo      -- shape (6,): lower bound (used to clip the result in-range)
      lat_hi      -- shape (6,): upper bound (used to clip the result in-range)

    OUTPUT:
      shape (6,): a new latent vector close to best_latent, clipped to valid bounds
    """
    step = np.random.normal(0.0, 1.0, size=best_latent.shape).astype(float)
    z    = np.asarray(best_latent, dtype=float) + float(lr) * step
    # Clip to valid bounds so local proposals don't escape the decoder's range
    return np.clip(z, lat_lo, lat_hi).astype(float)


# ===========================================================================
# STEP 3: Smart seed search -- start from a real training foil
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
    PURPOSE:
      Before the main optimization loop, try a random sample of REAL training
      airfoils to find a valid starting point (a foil that passes all hard
      geometry checks and has a finite CD/CL score).

    WHY THIS IS CRITICAL:
      In the old run, the first "best" foil was actually physically impossible
      (surfaces crossed, tmin < 0). Since 75% of proposals after that were
      LOCAL refinements around the "best", all 800 iterations explored the
      same invalid region. The output was a nonsense foil shape.

      By starting from a KNOWN VALID training foil, we guarantee that local
      refinement begins from somewhere good.

    HOW IT CONNECTS:
      Called once at the start of nom_optimize(), before the main loop.
      If a valid seed is found, it becomes the initial "best" for the loop.
      If no seed is found, the main loop starts with pure global exploration.

    INPUTS:
      pipeline       -- TalarAIPipeline (decoder + NeuralFoil)
      all_latents    -- shape (N, 6): full dataset of latent vectors
      alpha, Re      -- fixed operating conditions for this optimization run
      lat_lo, lat_hi -- bounds from latent_minmax_bounds()
      penalty_kwargs -- constraint parameters (same dict used in main loop)
      n_seed_tries   -- how many random training foils to try (default 100)

    OUTPUT:
      dict with keys {iter, mode, lr, CL, CD, objective, penalty, total,
                      tmin_int, tmax_int, camber_max_abs_int, te_gap, le_gap,
                      latent, coords}
      or None if no valid seed was found.
    """
    n       = len(all_latents)
    # Pick n_seed_tries random foils from the training dataset (no repeats)
    indices = np.random.choice(n, size=min(n_seed_tries, n), replace=False)

    best_seed = None

    for idx in indices:
        cand = all_latents[idx].copy()

        # Evaluate through full pipeline
        res = safe_eval(pipeline, cand, alpha=alpha, Re=Re)
        if res is None:
            continue   # pipeline crashed or returned bad values -- skip

        CL, CD, coords = res["CL"], res["CD"], res["coords"]

        # Score the candidate
        obj = default_objective(CL, CD)

        pen, pen_info = total_penalty(
            latent_vec=cand, coords=coords, CL=CL,
            lat_lo=lat_lo, lat_hi=lat_hi, **penalty_kwargs
        )

        # Skip hard rejects and non-finite objectives
        if not (np.isfinite(pen) and np.isfinite(obj)):
            continue

        total = float(obj + pen)

        # Keep the seed with the lowest total score
        if best_seed is None or total < best_seed["total"]:
            best_seed = {
                "iter": 0,
                "mode": "seed",
                "lr":   0.0,
                "CL":   float(CL),
                "CD":   float(CD),
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
# MAIN NOM OPTIMIZATION LOOP
# ===========================================================================

def nom_optimize(
    *,
    # --- Operating conditions (fixed for the entire run) ---
    alpha: float = 6.0,    # Angle of attack in degrees
                            # 6° is a reasonable cruise AoA for hydrofoils
    Re:    float = 5e5,    # Reynolds number (Re = V * chord / nu)
                            # 5e5 corresponds to 1/15 scale model test conditions

    # --- How many iterations to run ---
    n_iters: int = 3000,   # Total number of candidate foils to try.
                            # Higher = more exploration, longer runtime.
                            # 3000 is enough to converge in most cases.

    # --- Learning rate for local proposals ---
    learning_rate_init: float = 5e-3,   # Initial step size for local nudges.
                                         # 5e-3 gives bigger early steps so the
                                         # optimizer actually explores after seeding.
    lr_decay:           float = 0.999,  # Multiply lr by this each iteration.
                                         # 0.999 means lr shrinks slowly so
                                         # early steps explore, late steps refine.

    # --- Global vs local proposal balance ---
    # p_local: fraction of iterations that use local refinement (vs global).
    # 0.75 = 75% local, 25% global after a valid seed is found.
    # IMPORTANT: If no valid best exists yet, we always use global (100%)
    # to explore broadly before committing to local refinement.
    p_local: float = 0.75,

    # --- Lambda weights for the penalty terms (from constraints.py) ---
    # These control how strongly each constraint pulls against the CD/CL objective.
    # Think of them as "how many units of CD/CL improvement does it take to
    # justify violating this constraint by 1 unit?"
    lam_bounds: float = 1.0,    # Penalty weight for latent out-of-bounds
                                  # (keeps optimizer in decoder's trained range)
    lam_geom:   float = 25.0,   # Penalty weight for soft geometry violations
                                  # (TE/LE gaps). Hard violations always return inf.
    lam_cl:     float = 50.0,   # Penalty weight for CL outside designer's window
                                  # (enforces minimum lift and avoids cavitation)
                                  # Raised from 10 -> 50 so CL violations actually
                                  # hurt enough to push the optimizer back in bounds

    # --- Geometry hard limits ---
    # These define what counts as a "real" foil shape. Violations are hard rejects.
    min_thickness:  float = 0.04,   # Minimum allowed foil thickness (chord fraction)
                                     # Set from the MINIMUM thickness observed across
                                     # ALL training airfoils in the dataset.
    max_thickness:  float = 0.14,   # Maximum allowed thickness (chord fraction)
                                     # Prevents unrealistic "blob" shapes.
    camber_max_abs: float = 0.05,   # Maximum mean camber line deviation
                                     # Real airfoils rarely exceed ~0.12 camber.
    te_gap_max:     float = 0.01,   # Max allowed trailing edge gap (soft penalty)
    le_gap_max:     float = 0.01,   # Max allowed leading edge gap (soft penalty)

    # --- CL operating window ---
    # The foil must generate enough lift (cl_min) without risking cavitation (cl_max).
    # Set to None to disable either bound (e.g., cl_min=None means any CL is ok).
    cl_min: float | None = 0.30,   # Minimum CL -- foil must generate this much lift
    cl_max: float | None = 0.85,   # Maximum CL -- above this risks cavitation
                                    # Raised from 0.70: previous run best CL=0.745
                                    # was being penalized for exceeding 0.70, causing
                                    # optimizer to diverge. 0.85 gives real headroom.

    # --- Seed search: how many real training foils to try before the main loop ---
    n_seed_tries: int = 200,   # Number of dataset foils to test before starting.
                                # Higher = better chance of a good starting point.

    out_dir: str = "outputs",   # Directory where all output files are saved
):
    """
    Run the full NOM (Neural Optimization Machine) optimization loop and
    save the best foil shape found.

    OUTPUTS SAVED TO out_dir/:
      best_latent_nom.npy    -- the 6 best latent parameters as numpy array
      best_latent_nom.csv    -- same, in CSV format (for use in other scripts)
      best_coords_nom.csv    -- the 80x2 foil coordinates of the best shape
      nom_history.json       -- log of every valid iteration (CL, CD, scores)
      nom_summary.json       -- final summary (best CL, CD, L/D, constraints used)

    HOW THE LOOP WORKS:
      iter 1..n_iters:
        1. Propose candidate latent vector (global or local)
        2. Evaluate: latent -> decoder -> coords -> NeuralFoil -> CL, CD
        3. If pipeline fails (safe_eval returns None): skip, continue
        4. Compute objective = CD/CL
        5. Compute penalty   = total_penalty() from constraints.py
        6. If penalty = inf (hard geometry reject): skip, continue
        7. If obj + penalty < best so far: update best, print progress
        8. Decay learning rate: lr *= lr_decay
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load the training dataset and compute latent parameter bounds
    # -----------------------------------------------------------------------
    # These bounds define the valid search box for the optimizer.
    # They come from the actual min/max of each parameter across all training foils.
    latents         = load_latent_dataset()
    lat_lo, lat_hi  = latent_minmax_bounds(latents)

    print("\n=== DATASET LATENT PARAMETER BOUNDS (min/max across ALL training foils) ===")
    for i in range(6):
        print(f"  p{i+1}: min = {lat_lo[i]: .6f},  max = {lat_hi[i]: .6f},  "
              f"range = {lat_hi[i] - lat_lo[i]: .6f}")
    print("============================================================================\n")

    # Initialize the decoder + NeuralFoil pipeline
    pipeline = TalarAIPipeline()

    # Package all constraint parameters into one dict for easy reuse.
    # This dict is passed into total_penalty() every iteration.
    penalty_kwargs = dict(
        lam_bounds=lam_bounds,
        lam_geom=lam_geom,
        lam_cl=lam_cl,
        min_thickness=min_thickness,
        max_thickness=max_thickness,
        camber_max_abs=camber_max_abs,
        te_gap_max=te_gap_max,
        le_gap_max=le_gap_max,
        cl_min=cl_min,
        cl_max=cl_max,
    )

    # -----------------------------------------------------------------------
    # Seed search: try actual training foils to get a valid starting point
    # -----------------------------------------------------------------------
    print(f"Seed search: testing {n_seed_tries} random training airfoils first...")
    best = find_valid_seed(
        pipeline, latents,
        alpha=alpha, Re=Re,
        lat_lo=lat_lo, lat_hi=lat_hi,
        penalty_kwargs=penalty_kwargs,
        n_seed_tries=n_seed_tries,
    )

    if best is not None:
        print(
            f"Valid seed found! "
            f"total={best['total']:.5f} "
            f"(obj={best['objective']:.5f}, pen={best['penalty']:.5f}) | "
            f"CL={best['CL']:.4f}  CD={best['CD']:.6f} | "
            f"tmin={best['tmin_int']:.4f}  tmax={best['tmax_int']:.4f}  "
            f"camber={best['camber_max_abs_int']:.4f}"
        )
    else:
        print("WARNING: No valid seed found from training dataset foils.")
        print("         Starting main loop with 100% global exploration.")
        print("         If the main loop also finds nothing, try:")
        print("           cl_min=None, cl_max=None   (remove CL constraint)")
        print("           max_thickness=0.25")
        print("           camber_max_abs=0.20")

    # -----------------------------------------------------------------------
    # Main NOM optimization loop
    # -----------------------------------------------------------------------
    history = []   # list of dicts: one entry per valid (non-rejected) iteration
    valid   = 0    # count of iterations where the candidate passed all checks
    skipped = 0    # count of iterations where the candidate was rejected
    lr      = float(learning_rate_init)   # current learning rate (decays each iter)

    for it in range(1, n_iters + 1):

        # --- PROPOSE: decide global or local ---
        # If no valid best yet: always global (broad random search)
        # If valid best exists: p_local fraction of time go local (refine best)
        use_local = (best is not None) and (np.random.rand() < float(p_local))

        if use_local:
            # LOCAL: nudge the current best latent by a small random step
            cand = propose_local(best["latent"], lr=lr, lat_lo=lat_lo, lat_hi=lat_hi)
            mode = "local"
        else:
            # GLOBAL: completely random point in the valid latent bounds box
            cand = propose_global(lat_lo, lat_hi)
            mode = "global"

        # --- EVALUATE: run through decoder + NeuralFoil ---
        res = safe_eval(pipeline, cand, alpha=alpha, Re=Re)
        if res is None:
            # Pipeline failed (bad latent, NeuralFoil crash, etc.) -- skip
            skipped += 1
            lr *= float(lr_decay)   # still decay lr to keep pacing consistent
            continue

        CL, CD, coords = res["CL"], res["CD"], res["coords"]

        # --- OBJECTIVE: CD/CL (we want to minimize this = maximize L/D) ---
        obj = default_objective(CL, CD)

        # --- PENALTY: physical and geometric constraints ---
        pen, pen_info = total_penalty(
            latent_vec=cand, coords=coords, CL=CL,
            lat_lo=lat_lo, lat_hi=lat_hi, **penalty_kwargs
        )

        # --- HARD REJECT: skip if penalty or objective is infinite ---
        # pen = inf means a hard geometry violation (crossing, too thin/thick, camber)
        # obj = inf means CL <= 0 (foil generates no positive lift)
        # Either way, this candidate can NEVER be accepted as the best result.
        if not (np.isfinite(pen) and np.isfinite(obj)):
            skipped += 1
            lr *= float(lr_decay)
            continue

        # --- This candidate is valid -- count it and score it ---
        valid += 1
        total = float(obj + pen)

        # Record this iteration in the history log
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

        # --- UPDATE BEST: if this is the best total score so far, keep it ---
        if best is None or total < best["total"]:
            best = {**rec, "latent": cand.copy(), "coords": coords.copy()}
            print(
                f"[{it:4d}/{n_iters}] NEW BEST | "
                f"total={best['total']:.5f} "
                f"(obj={best['objective']:.5f}, pen={best['penalty']:.5f}) | "
                f"CL={best['CL']:.4f}  CD={best['CD']:.6f} | "
                f"tmin={best['tmin_int']:.4f}  tmax={best['tmax_int']:.4f}  "
                f"camber={best['camber_max_abs_int']:.4f} | "
                f"lr={best['lr']:.2e}  mode={best['mode']}"
            )

        # --- DECAY learning rate each iteration (local steps get smaller over time) ---
        lr *= float(lr_decay)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    if best is None:
        print("\n" + "=" * 60)
        print("NOM found 0 valid candidates after all checks.")
        print("Try loosening these parameters in nom_optimize():")
        print("  cl_min=None, cl_max=None     (remove the CL window constraint)")
        print("  max_thickness=0.30           (allow thicker foils)")
        print("  camber_max_abs=0.20          (allow more camber)")
        print("  n_iters=5000                 (more iterations)")
        print("=" * 60)
        return

    # Save the best latent vector (the 6 optimized parameters)
    np.save(out_path / "best_latent_nom.npy", best["latent"])
    np.savetxt(
        out_path / "best_latent_nom.csv",
        best["latent"].reshape(1, -1),
        delimiter=",",
        header="p1,p2,p3,p4,p5,p6",
        comments=""
    )

    # Save the best foil coordinates (80 x 2 array of [x, y] points)
    np.savetxt(
        out_path / "best_coords_nom.csv",
        best["coords"],
        delimiter=",",
        header="x,y",
        comments=""
    )

    # Save the full iteration history (for plotting convergence)
    with open(out_path / "nom_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Save a human-readable summary of this run
    summary = {
        "alpha":               float(alpha),
        "Re":                  float(Re),
        "n_iters":             int(n_iters),
        "learning_rate_init":  float(learning_rate_init),
        "lr_decay":            float(lr_decay),
        "p_local":             float(p_local),
        "lam_bounds":          float(lam_bounds),
        "lam_geom":            float(lam_geom),
        "lam_cl":              float(lam_cl),
        "min_thickness":       float(min_thickness),
        "max_thickness":       float(max_thickness),
        "camber_max_abs":      float(camber_max_abs),
        "te_gap_max":          float(te_gap_max),
        "le_gap_max":          float(le_gap_max),
        "cl_min":              None if cl_min is None else float(cl_min),
        "cl_max":              None if cl_max is None else float(cl_max),
        "n_seed_tries":        int(n_seed_tries),
        "valid_evals":         int(valid),
        "skipped":             int(skipped),
        "best_total":          float(best["total"]),
        "best_objective":      float(best["objective"]),
        "best_penalty":        float(best["penalty"]),
        "best_CL":             float(best["CL"]),
        "best_CD":             float(best["CD"]),
        "best_latent_params":  [float(x) for x in best["latent"].reshape(-1)],
        "latent_lo":           [float(x) for x in lat_lo],
        "latent_hi":           [float(x) for x in lat_hi],
    }

    with open(out_path / "nom_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== NOM Optimization Complete ===")
    print(f"  Best CL   = {best['CL']:.4f}")
    print(f"  Best CD   = {best['CD']:.6f}")
    print(f"  Best L/D  = {best['CL'] / best['CD']:.2f}")
    print(f"  CD/CL     = {best['objective']:.5f}")
    print(f"  Valid evals: {valid} / {n_iters}  (skipped: {skipped})")
    print(f"  All outputs saved to: {out_path}/")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    nom_optimize()