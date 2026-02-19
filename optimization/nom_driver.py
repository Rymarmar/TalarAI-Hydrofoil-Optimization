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
import json
from pathlib import Path

import numpy as np
import pandas as pd

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
      json_path -- path to best_baseline_foil_alpha4_Re5e5.json
    
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
# MAIN NOM OPTIMIZATION
# ===========================================================================

def nom_optimize(
    *,
    # --- Operating conditions (PROF ACTION ITEM: alpha=4 instead of 6) ---
    alpha: float = 4.0,  # ACTION ITEM: "6 is too harsh" → changed to 4
    Re: float = 5e5,     # ACTION ITEM: "5e5 also for reynolds"
    
    # --- Iterations ---
    n_iters: int = 3000,
    
    # --- Learning rate ---
    learning_rate_init: float = 0.005,
    lr_decay: float = 0.999,
    
    # --- Strategy balance ---
    p_local: float = 0.75,
    
    # --- Lambda weights (AUTO-NORMALIZED in constraints.py) ---
    lam_bounds: float = 1.0,
    lam_geom: float = 25.0,
    lam_cl: float = 50.0,
    
    # --- Geometry limits (ACTION ITEM: removed camber, le_gap) ---
    min_thickness: float = 0.006,   # dataset min (0.0071) * 0.9; 0.04 rejects ALL training foils
    max_thickness: float = 0.157,  # dataset max (0.1427) * 1.1
    te_gap_max: float = 0.01,  # ACTION ITEM: only TE, no LE
    
    # --- CL window ---
    cl_min: float | None = 0.5,
    cl_max: float | None = 1.6,   # was 0.85 -- too tight! best foil has CL=1.39
    
    # --- Paths ---
    csv_path: str = "data/airfoil_latent_params_6.csv",
    lookup_baseline_path: str = "outputs/best_baseline_foil_alpha4_Re5e+05.json",
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
    print(f"Strategy: {p_local*100:.0f}% local, {(1-p_local)*100:.0f}% global")
    print("=" * 70)
    print()
    
    # -----------------------------------------------------------------------
    # Load dataset and compute latent bounds
    # -----------------------------------------------------------------------
    
    print("Loading training dataset...")
    all_latents = load_latent_dataset(csv_path)
    lat_lo, lat_hi = latent_minmax_bounds(all_latents)
    
    print(f"✓ Loaded {len(all_latents)} foils")
    print(f"  Latent bounds:")
    for i in range(6):
        print(f"    p{i+1}: [{lat_lo[i]:+.4f}, {lat_hi[i]:+.4f}]")
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
    
    baseline = load_best_baseline(lookup_baseline_path)
    
    best = None
    
    if baseline is not None:
        print(f"Baseline foil: {baseline['filename']}")
        print(f"  CL:        {baseline['CL']:.4f}")
        print(f"  CD:        {baseline['CD']:.6f}")
        print(f"  L/D:       {baseline['L_over_D']:.2f}")
        print(f"  CD/CL:     {baseline['CD_over_CL']:.6f}")
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
            }
            
            pen, pen_info = total_penalty(
                latent_vec=latent_baseline,
                coords=coords,
                CL=CL,
                lat_lo=lat_lo,
                lat_hi=lat_hi,
                **penalty_kwargs
            )
            
            if np.isfinite(pen) and np.isfinite(obj):
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
    }
    
    history = []
    valid = 0
    skipped = 0
    lr = float(learning_rate_init)
    
    # Diagnostic counters (printed for first 20 skips to help debug)
    skip_reasons = {"safe_eval_none": 0, "pen_ge_1000": 0, "nonfinite": 0}
    diag_printed = 0  # how many skip diagnostics we've printed
    
    for it in range(1, n_iters + 1):
        
        # --- PROPOSE ---
        use_local = (best is not None) and (np.random.rand() < float(p_local))
        
        if use_local:
            cand = propose_local(best['latent'], lr=lr, lat_lo=lat_lo, lat_hi=lat_hi)
            mode = "local"
        else:
            cand = propose_global(lat_lo, lat_hi)
            mode = "global"
        
        # --- EVALUATE ---
        res = safe_eval(pipeline, cand, alpha=alpha, Re=Re)
        if res is None:
            skip_reasons["safe_eval_none"] += 1
            skipped += 1
            if diag_printed < 5:
                # Run again with debug=True to see why
                safe_eval(pipeline, cand, alpha=alpha, Re=Re, debug=True)
                diag_printed += 1
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
        if not (np.isfinite(pen) and np.isfinite(obj)) or pen >= 1000.0:
            skip_reasons["pen_ge_1000"] += 1
            skipped += 1
            if diag_printed < 10:
                print(f"  [diag] iter={it} HARD REJECT pen={pen:.2f} obj={obj:.4f} "
                      f"CL={CL:.4f} reason={pen_info.get('reason','?')} "
                      f"t_min={pen_info.get('t_min',0):.4f} t_max={pen_info.get('t_max',0):.4f}")
                diag_printed += 1
            lr *= float(lr_decay)
            continue
        
        # --- VALID CANDIDATE ---
        valid += 1
        total = float(obj + pen)
        
        rec = {
            'iter': int(it),
            'mode': mode,
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
            print(
                f"[{it:4d}/{n_iters}] NEW BEST | "
                f"total={total:.6f} "
                f"(obj={obj:.6f}, pen={pen:.6f}) | "
                f"CL={CL:.4f} CD={CD:.6f} | "
                f"L/D={CL/CD:.1f} | "
                f"tmin={pen_info.get('t_min', 0):.4f} tmax={pen_info.get('t_max', 0):.4f} | "
                f"lr={lr:.2e} {mode}"
            )
        
        # --- DECAY LR ---
        lr *= float(lr_decay)
    
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
    print(f"  → safe_eval returned None: {skip_reasons['safe_eval_none']}")
    print(f"  → hard reject (pen≥1000):  {skip_reasons['pen_ge_1000']}")
    print(f"Outputs:   {out_path}/")
    print("=" * 70)


if __name__ == "__main__":
    nom_optimize()