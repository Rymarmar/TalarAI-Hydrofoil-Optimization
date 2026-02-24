"""
tools/build_lookup_table.py

WHAT THIS FILE DOES (plain English):
  We have 1647 real airfoils from the UIUC database that we used to train
  the encoder/decoder. Each one has been compressed into 6 latent numbers
  (p1..p6) by the encoder.

  This script asks: "Of all 1647 training foils, which one performs best
  at our actual operating conditions?" That winner becomes the STARTING
  POINT for the NOM optimizer.

WHY WE NEED A STARTING POINT:
  NOM works by taking small steps in latent space (the 6-number space).
  If you start from a bad foil, you waste hundreds of iterations just
  getting to a reasonable region. If you start from the BEST foil in
  the training set, every step NOM takes is already an improvement on
  a strong baseline. This is the professor's action item:
  "Use lookup table instead of random seeds."

WHY 12 CONDITIONS (6 alpha x 2 Re):
  Your hydrofoil doesn't only operate at one speed/angle. It has to:
    - Lift off the water (slow speed, high angle of attack)
    - Cruise at design speed (alpha~1 deg, Re~440k)
    - Operate at max test speed (alpha~1 deg, Re~440k)

  If we only evaluate foils at one condition, we might pick a foil that
  is great at max speed but terrible at takeoff. By averaging L/D across
  all 12 conditions we find the foil that is best all-around across the
  real operating envelope of the SkiCat.

HOW THE PIPELINE WORKS HERE (same as NOM):
  raw .dat file → encoder → 6 latent params (already done, saved in CSV)
  6 latent params → decoder → 80 coords → NeuralFoil → CL, CD, L/D

  IMPORTANT: we use the DECODER to reconstruct each foil, not the raw
  .dat file directly. This is intentional -- NOM can only work with foils
  the decoder can produce, so we evaluate foils the same way NOM will.
  Using raw .dat files would give different L/D values than NOM sees,
  making the baseline comparison inconsistent.

OUTPUTS (per condition):
  lookup_table_alpha{a}_Re{Re}.csv      -- all 1647 foils ranked by L/D
  top_100_foils_alpha{a}_Re{Re}.csv     -- top 100 only
  best_baseline_foil_alpha{a}_Re{Re}.json -- best single foil for nom_driver

OUTPUTS (averaged across all 12 conditions):
  lookup_table_averaged_all_conditions.csv  -- all 1647 ranked by mean L/D
  best_baseline_foil_averaged.json          -- best all-around foil

HOW TO RUN:
  python tools/build_lookup_table.py

  Runtime: ~20 minutes (12 conditions x 1647 foils).
  Run once, results are saved. Re-run only if alpha/Re grid changes.
"""

from __future__ import annotations
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from pipeline.talarai_pipeline import TalarAIPipeline
except ModuleNotFoundError:
    from talarai_pipeline import TalarAIPipeline


# ============================================================================
# CONFIGURATION
# ============================================================================

LATENT_CSV = PROJECT_ROOT / "data" / "airfoil_latent_params_6.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 6x2 = 12 conditions total.
#
# ALPHA range (degrees):
#   Physical basis: from the Ski Cat Info spreadsheet, the full alpha
#   operating range is 0 to 15 degrees. We sweep 0.5 to 4.0 which covers
#   the realistic in-flight range -- takeoff transition through cruise.
#   We stop at 4.0 because thin cambered foils approach stall around 8-10
#   degrees and NeuralFoil predictions become unreliable above that.
#   Design point is alpha ~ 1 deg at max speed (from CL requirement slides:
#   alpha = CL/2pi = 0.111/2pi = 1.012 deg).
#
# RE range (Reynolds number, dimensionless):
#   Physical basis: Re = V * chord / nu_water
#   chord = 2.25 in = 0.1875 ft
#   nu_water = 1.08e-5 ft^2/s (kinematic viscosity of water)
#
#   From Ski Cat Info spreadsheet (1/16 scale model):
#     Slow speed:  V = 8.44  ft/s  --> Re = 8.44  * 0.1875 / 1.08e-5 = 146,528 ~ 1.5e5
#     Max speed:   V = 25.32 ft/s  --> Re = 25.32 * 0.1875 / 1.08e-5 = 439,583 ~ 4.4e5
#
#   We use 150,000 and 440,000 to match the spreadsheet values exactly.
#   NOTE: these are WATER Reynolds numbers. Air Re values (50k-200k seen
#   in XFLR5 / wind tunnel references) are NOT applicable here.

ALPHA_LIST = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]  # degrees -- 1.0 is design point

# ACTION ITEM (build_lookup_table 2/19): Four Re values sweeping the operating envelope.
# Correct values confirmed: 150000, 250000, 350000, 450000
# These are WATER Reynolds numbers (Re = V * chord / nu_water) for the 1/15 scale model.
# Spans from low-speed / takeoff up to max test speed in the Davidson Lab towing tank.
RE_LIST = [150000, 250000, 350000, 450000]   # ACTION ITEM (build_lookup_table 2/19)


# ============================================================================
# SINGLE-CONDITION EVALUATION
# ============================================================================

def evaluate_condition(pipeline, all_latents, filenames, alpha, Re):
    """
    Evaluate all 1647 foils at one (alpha, Re) pair.

    For each foil:
      1. Take its 6 latent params from the training CSV
      2. Run through decoder to get 80 coordinate points
      3. Feed those coords into NeuralFoil to get CL and CD
      4. Compute L/D = CL/CD (higher is better)

    Returns a DataFrame of all 1647 foils sorted best-first by L/D.
    Foils where NeuralFoil fails (bad geometry, overflow) get L/D = NaN
    and sink to the bottom of the ranking.
    """
    results = []
    n = len(all_latents)

    for i, (latent, filename) in enumerate(zip(all_latents, filenames)):
        if (i + 1) % 200 == 0 or i == 0:
            print(f"    [{i+1:4d}/{n}] {filename}...")
        try:
            aero = pipeline.eval_latent_with_neuralfoil(
                latent_vec=latent,
                alpha=alpha,
                Re=Re,
                model_size="xlarge",
                debug=False,
            )
            CL = float(aero.get("CL", np.nan))
            CD = float(aero.get("CD", np.nan))

            if np.isfinite(CL) and np.isfinite(CD) and CL > 0 and CD > 0:
                L_over_D   = CL / CD
                CD_over_CL = CD / CL
            else:
                L_over_D = CD_over_CL = np.nan

        except Exception:
            CL = CD = L_over_D = CD_over_CL = np.nan

        results.append({
            'filename':   filename,
            'CL':         CL,
            'CD':         CD,
            'L_over_D':   L_over_D,
            'CD_over_CL': CD_over_CL,
            'p1': float(latent[0]),
            'p2': float(latent[1]),
            'p3': float(latent[2]),
            'p4': float(latent[3]),
            'p5': float(latent[4]),
            'p6': float(latent[5]),
        })

    df = pd.DataFrame(results)
    df = df.sort_values('L_over_D', ascending=False, na_position='last')
    return df


# ============================================================================
# CAMBER FILTER UTILITIES
# ============================================================================

def compute_max_camber(coords: np.ndarray,
                       x_min: float = 0.05,
                       x_max: float = 0.90) -> float:
    """
    Compute the maximum absolute camber of a foil from its (80,2) coords.

    WHAT IS CAMBER:
      The camber line is the midpoint between upper and lower surfaces at each x.
      For a symmetric NACA 00xx foil, camber = 0 everywhere.
      For a cambered foil (e.g. NACA 4412), camber peaks at ~4%c near midchord.

    INPUTS:
      coords -- shape (80, 2): same convention as talarai_pipeline output
        rows 0-39  = upper surface TE->LE (x: 1->0)
        rows 40-79 = lower surface LE->TE (x: 0->1)
      x_min, x_max -- interior range to check (avoids LE/TE noise)

    OUTPUT:
      max absolute camber as fraction of chord (e.g. 0.04 = 4%c)
      Returns 999.0 if coords are bad (will be filtered out)
    """
    if coords is None or coords.shape != (80, 2):
        return 999.0

    upper_te2le = coords[:40]
    lower_le2te = coords[40:]
    upper_le2te = upper_te2le[::-1]   # flip to LE->TE

    xu, yu = upper_le2te[:, 0], upper_le2te[:, 1]
    xl, yl = lower_le2te[:, 0], lower_le2te[:, 1]

    # Both surfaces share the same x-grid (linspace 0->1, 40 pts)
    # Camber = midpoint between upper and lower at each x
    camber_line = (yu + yl) / 2.0

    # Only check interior to avoid LE/TE noise
    mask = (xu >= x_min) & (xu <= x_max)
    if not np.any(mask):
        return 999.0

    return float(np.max(np.abs(camber_line[mask])))


def rebaseline_with_camber_filter(max_camber: float = 0.04):
    """
    ACTION ITEM (2/23): Re-pick best baseline JSONs from existing lookup CSVs,
    applying a camber filter so NOM starts from a manufacturable foil.

    WHY THIS EXISTS:
      The original best baseline (e61.png) has 7.4%c camber -- too cambered
      to 3D print easily. But re-running the full Phase 1 (2 hours) just to
      change which foil gets selected as 'best' is wasteful.

      This function reads the already-computed lookup_table_*.csv files,
      reconstructs each foil's coords via the decoder to measure camber,
      filters out foils above max_camber, and re-saves the best JSON files.

    USAGE:
      python tools/build_lookup_table.py --rebaseline
      python tools/build_lookup_table.py --rebaseline --max-camber 0.03

    INPUTS:
      max_camber -- hard camber limit (default 0.04 = 4%c)
        Blocks Eppler 61 class (7.4%c), allows NACA 4412 (4%c) and below.
    """
    print("=" * 60)
    print(f"REBASELINE WITH CAMBER FILTER (max_camber={max_camber:.2f} = {max_camber*100:.0f}%c)")
    print("Reading existing lookup CSVs -- no Phase 1 re-run needed.")
    print("=" * 60)
    print()

    # Load pipeline once for camber computation
    print("Loading TalarAI pipeline...")
    pipeline = TalarAIPipeline()
    print(f"Pipeline ready: {pipeline.decoder_path.name}")
    print()

    # Load latent params for coord reconstruction
    df_csv = pd.read_csv(LATENT_CSV)
    latent_cols = [c for c in df_csv.columns if c.lower().startswith('p')]
    all_latents = df_csv[latent_cols].values.astype(np.float32)
    filenames = df_csv['filename'].tolist() if 'filename' in df_csv.columns \
                else [f"foil_{i}" for i in range(len(df_csv))]

    # Pre-compute camber for every foil ONCE (shared across all conditions)
    print(f"Pre-computing camber for {len(all_latents)} foils...")
    camber_map = {}   # filename -> max_camber_value
    for i, (latent, fname) in enumerate(zip(all_latents, filenames)):
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(all_latents)}]  computing camber...")
        try:
            coords = pipeline.latent_to_coordinates(latent)
            camber_map[fname] = compute_max_camber(coords)
        except Exception:
            camber_map[fname] = 999.0

    n_pass = sum(1 for v in camber_map.values() if v <= max_camber)
    print(f"Foils passing camber filter: {n_pass}/{len(all_latents)} "
          f"({100*n_pass/len(all_latents):.1f}%)")
    print()

    # Re-pick best from each condition CSV
    existing_csvs = sorted(OUTPUT_DIR.glob("lookup_table_alpha*.csv"))
    existing_csvs = [f for f in existing_csvs if "averaged" not in f.name]

    if not existing_csvs:
        print(f"ERROR: No lookup_table_alpha*.csv files in {OUTPUT_DIR}")
        print("Run full build_lookup_table() first.")
        return

    rebaseline_summary = []
    for fpath in existing_csvs:
        tag = fpath.stem.replace("lookup_table_", "")
        df = pd.read_csv(fpath)

        if 'filename' not in df.columns or 'L_over_D' not in df.columns:
            print(f"  WARNING: {fpath.name} missing columns, skipping")
            continue

        # Apply camber filter: keep only foils whose camber is below threshold
        df['max_camber'] = df['filename'].map(camber_map).fillna(999.0)
        df_filtered = df[df['max_camber'] <= max_camber].copy()
        df_valid    = df_filtered.dropna(subset=['L_over_D'])
        df_valid    = df_valid.sort_values('L_over_D', ascending=False)

        if df_valid.empty:
            print(f"  WARNING: {tag} -- no valid foils after camber filter! "
                  f"Try raising max_camber.")
            continue

        best = df_valid.iloc[0]

        # Parse alpha and Re from tag (e.g. "alpha1.0_Re1.5e+05")
        try:
            parts   = tag.split('_Re')
            alpha_v = float(parts[0].replace('alpha', ''))
            Re_v    = float(parts[1])
        except Exception:
            alpha_v = Re_v = 0.0

        best_dict = {
            'filename':   best['filename'],
            'alpha':      alpha_v,
            'Re':         Re_v,
            'CL':         float(best['CL']),
            'CD':         float(best['CD']),
            'L_over_D':   float(best['L_over_D']),
            'CD_over_CL': float(best['CD_over_CL']),
            'max_camber':       float(best['max_camber']),
            'camber_filter_applied': max_camber,
            'latent': [float(best[f'p{i}']) for i in range(1, 7)],
        }
        best_file = OUTPUT_DIR / f"best_baseline_foil_{tag}.json"
        with open(best_file, 'w') as f:
            json.dump(best_dict, f, indent=2)

        print(f"  {tag}:")
        print(f"    Best foil: {best['filename']}  "
              f"L/D={best['L_over_D']:.1f}  "
              f"camber={best['max_camber']*100:.1f}%c")

        rebaseline_summary.append({
            'condition': tag,
            'best_foil': best['filename'],
            'L_over_D':  round(float(best['L_over_D']), 1),
            'camber_pct': round(float(best['max_camber']) * 100, 1),
        })

    print()
    print("=" * 60)
    print("REBASELINE COMPLETE")
    print(f"All best_baseline_foil_*.json files updated with camber <= {max_camber*100:.0f}%c")
    print("=" * 60)
    if rebaseline_summary:
        print(pd.DataFrame(rebaseline_summary).to_string(index=False))


# ============================================================================
# MAIN
# ============================================================================

def build_lookup_table():

    # Load the 1647 latent parameter vectors (p1..p6) from training CSV.
    # These were produced by running all UIUC foils through the encoder.
    if not LATENT_CSV.exists():
        raise FileNotFoundError(f"Latent CSV not found: {LATENT_CSV}")
    df_csv = pd.read_csv(LATENT_CSV)
    latent_cols = [c for c in df_csv.columns if c.lower().startswith('p')]
    all_latents = df_csv[latent_cols].values.astype(np.float32)
    filenames   = df_csv['filename'].tolist() if 'filename' in df_csv.columns \
                  else [f"foil_{i}" for i in range(len(df_csv))]
    print(f"Loaded {len(all_latents)} foils from {LATENT_CSV.name}")

    # Load the decoder pipeline ONCE and reuse it for all 12 conditions.
    # Loading the TF model is expensive (~5 sec), evaluating each foil is cheap.
    print("Loading TalarAI pipeline (decoder + NeuralFoil)...")
    pipeline = TalarAIPipeline()
    print(f"Pipeline ready: {pipeline.decoder_path.name}")
    print(f"Running {len(ALPHA_LIST)} x {len(RE_LIST)} = "
          f"{len(ALPHA_LIST)*len(RE_LIST)} conditions...\n")

    total_conditions = len(ALPHA_LIST) * len(RE_LIST)
    done = 0
    summary_rows = []  # collects best-foil info per condition for final printout

    # ------------------------------------------------------------------
    # PHASE 1: Evaluate all foils at each (alpha, Re) condition
    # ------------------------------------------------------------------
    for alpha in ALPHA_LIST:
        for Re in RE_LIST:
            done += 1
            tag = f"alpha{alpha:.1f}_Re{Re:.0e}"
            print("=" * 60)
            print(f"[{done}/{total_conditions}]  alpha={alpha}deg  Re={Re:.0e}")
            print("=" * 60)

            df = evaluate_condition(pipeline, all_latents, filenames, alpha, Re)

            # Save full ranking for this condition (all 1647 foils)
            lookup_file = OUTPUT_DIR / f"lookup_table_{tag}.csv"
            df.to_csv(lookup_file, index=False)
            print(f"  Saved full table  -> {lookup_file.name}")

            # Save top 100 only (useful for quick inspection)
            df_valid = df.dropna(subset=['L_over_D'])
            top100_file = OUTPUT_DIR / f"top_100_foils_{tag}.csv"
            df_valid.head(100).to_csv(top100_file, index=False)
            print(f"  Saved top 100     -> {top100_file.name}")

            if df_valid.empty:
                print("  WARNING: no valid foils at this condition, skipping JSON")
                continue

            # Save the single best foil as a JSON.
            # nom_driver.py loads this JSON to get its starting latent vector.
            # The JSON includes the latent params so NOM can begin stepping
            # from this foil's position in latent space immediately.
            best = df_valid.iloc[0]
            best_dict = {
                'filename':   best['filename'],
                'alpha':      alpha,
                'Re':         Re,
                'CL':         float(best['CL']),
                'CD':         float(best['CD']),
                'L_over_D':   float(best['L_over_D']),
                'CD_over_CL': float(best['CD_over_CL']),
                'latent': [float(best[f'p{i}']) for i in range(1, 7)],
            }
            best_file = OUTPUT_DIR / f"best_baseline_foil_{tag}.json"
            with open(best_file, 'w') as f:
                json.dump(best_dict, f, indent=2)
            print(f"  Saved best foil   -> {best_file.name}")
            print(f"  Best: {best['filename']}  "
                  f"L/D={best['L_over_D']:.1f}  "
                  f"CL={best['CL']:.4f}  CD={best['CD']:.6f}")

            summary_rows.append({
                'alpha':     alpha,
                'Re':        Re,
                'best_foil': best['filename'],
                'L_over_D':  round(float(best['L_over_D']), 1),
                'CL':        round(float(best['CL']), 4),
                'CD':        round(float(best['CD']), 6),
            })
            print()

    # ------------------------------------------------------------------
    # PHASE 2: Average L/D across all 12 conditions per foil.
    #
    # WHY: A foil that ranks #1 at alpha=1 might rank #50 at alpha=3.
    # The mean L/D finds the foil that is consistently good everywhere,
    # not just optimal at one point. This is the better baseline for NOM
    # because the real SkiCat operates across both slow and max speed.
    #
    # min_L_over_D is also saved -- a foil with high mean but very low
    # minimum has a weak spot in the envelope, which is worth knowing.
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 2: AVERAGING L/D ACROSS ALL 12 CONDITIONS")
    print("=" * 60)

    all_dfs = []
    for alpha in ALPHA_LIST:
        for Re in RE_LIST:
            tag = f"alpha{alpha:.1f}_Re{Re:.0e}"
            fpath = OUTPUT_DIR / f"lookup_table_{tag}.csv"
            if fpath.exists():
                df_c = pd.read_csv(fpath)[
                    ['filename', 'L_over_D', 'p1','p2','p3','p4','p5','p6']
                ]
                df_c = df_c.rename(columns={'L_over_D': f'LD_{tag}'})
                all_dfs.append(df_c)

    if all_dfs:
        # Merge all per-condition tables into one wide table.
        # Each row = one foil, each LD_ column = its L/D at one condition.
        #
        # FIX (2/19 run): ld_cols is now collected AFTER the merge loop by
        # scanning df_avg for columns that start with 'LD_'. Previously it was
        # built during the loop, which caused a KeyError when pandas renamed
        # columns due to collisions (adding _x/_y suffixes). Reading from the
        # actual DataFrame columns after merging is always correct regardless
        # of what Re/alpha tags were used or how pandas formats the names.
        df_avg = all_dfs[0][['filename','p1','p2','p3','p4','p5','p6']].copy()
        for df_c in all_dfs:
            ld_col = [c for c in df_c.columns if c.startswith('LD_')][0]
            df_avg = df_avg.merge(
                df_c[['filename', ld_col]], on='filename', how='left'
            )

        # Collect ld_cols from what's actually in df_avg after all merges
        ld_cols = [c for c in df_avg.columns if c.startswith('LD_')]
        print(f"  Averaging across {len(ld_cols)} conditions: {ld_cols}")

        # Compute mean and min across all conditions.
        # skipna=True means a foil that failed at 1 condition out of 12
        # is still ranked based on its 11 valid results.
        df_avg['mean_L_over_D'] = df_avg[ld_cols].mean(axis=1, skipna=True)
        df_avg['min_L_over_D']  = df_avg[ld_cols].min(axis=1, skipna=True)
        df_avg['n_valid']       = df_avg[ld_cols].notna().sum(axis=1)

        df_avg = df_avg.sort_values(
            'mean_L_over_D', ascending=False, na_position='last'
        )

        # Save full averaged ranking
        avg_file = OUTPUT_DIR / "lookup_table_averaged_all_conditions.csv"
        df_avg.to_csv(avg_file, index=False)
        print(f"  Saved averaged ranking -> {avg_file.name}")

        # Save best all-around foil JSON for nom_driver.py
        best_avg = df_avg.dropna(subset=['mean_L_over_D']).iloc[0]
        best_avg_dict = {
            'filename':           best_avg['filename'],
            'mean_L_over_D':      float(best_avg['mean_L_over_D']),
            'min_L_over_D':       float(best_avg['min_L_over_D']),
            'n_conditions_valid': int(best_avg['n_valid']),
            'conditions_averaged': [
                {'alpha': a, 'Re': r}
                for a in ALPHA_LIST for r in RE_LIST
            ],
            'latent': [float(best_avg[f'p{i}']) for i in range(1, 7)],
        }
        best_avg_file = OUTPUT_DIR / "best_baseline_foil_averaged.json"
        with open(best_avg_file, 'w') as f:
            json.dump(best_avg_dict, f, indent=2)
        print(f"  Saved best all-around  -> {best_avg_file.name}")
        print()
        print("  TOP 5 ALL-AROUND FOILS (by mean L/D across all 12 conditions):")
        print(df_avg[
            ['filename', 'mean_L_over_D', 'min_L_over_D', 'n_valid']
        ].head(5).to_string(index=False))

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("PER-CONDITION SUMMARY (best foil at each alpha/Re)")
    print("=" * 60)
    if summary_rows:
        print(pd.DataFrame(summary_rows).to_string(index=False))
    print()
    print("HOW TO USE IN nom_driver.py:")
    print("  Default (auto-loads per-condition best):")
    print("    nom_optimize(alpha=1.0, Re=15000)")
    print()
    print("  Use best all-around foil (averaged baseline):")
    print("    nom_optimize(alpha=1.0, Re=15000,")
    print("      lookup_baseline_path='outputs/best_baseline_foil_averaged.json')")
    print("=" * 60)


def run_phase2_averaging_only():
    """
    FIX (2/19 run): Run ONLY Phase 2 (averaging) without re-running Phase 1.

    If Phase 1 already completed successfully (all lookup_table_*.csv files
    exist in outputs/) but Phase 2 crashed, call this instead of build_lookup_table()
    to avoid waiting 2 hours again.

    Usage:
        python tools/build_lookup_table.py --phase2
    """
    print("=" * 60)
    print("PHASE 2 ONLY: AVERAGING L/D ACROSS ALL CONDITIONS")
    print("(Skipping Phase 1 -- reading existing lookup CSVs from outputs/)")
    print("=" * 60)

    # Scan outputs/ for whatever lookup_table_*.csv files actually exist,
    # regardless of what ALPHA_LIST/RE_LIST say. This is robust to any
    # Re/alpha combination and avoids the tag-mismatch KeyError.
    existing_csvs = sorted(OUTPUT_DIR.glob("lookup_table_alpha*.csv"))
    # Exclude the averaged output itself if it somehow exists
    existing_csvs = [f for f in existing_csvs if "averaged" not in f.name]

    if not existing_csvs:
        print(f"ERROR: No lookup_table_alpha*.csv files found in {OUTPUT_DIR}")
        print("Run the full build_lookup_table() first.")
        return

    print(f"Found {len(existing_csvs)} condition CSV(s):")
    for f in existing_csvs:
        print(f"  {f.name}")
    print()

    all_dfs = []
    for fpath in existing_csvs:
        tag = fpath.stem.replace("lookup_table_", "")  # e.g. "alpha1.0_Re1e+04"
        df_c = pd.read_csv(fpath)
        if 'L_over_D' not in df_c.columns or 'filename' not in df_c.columns:
            print(f"  WARNING: {fpath.name} missing required columns, skipping")
            continue
        needed = ['filename', 'L_over_D'] + [f'p{i}' for i in range(1, 7)]
        df_c = df_c[[c for c in needed if c in df_c.columns]]
        df_c = df_c.rename(columns={'L_over_D': f'LD_{tag}'})
        all_dfs.append(df_c)

    if not all_dfs:
        print("ERROR: No valid CSVs could be loaded.")
        return

    # Build base with latent params from first file
    first_cols = ['filename'] + [f'p{i}' for i in range(1, 7)]
    first_cols = [c for c in first_cols if c in all_dfs[0].columns]
    df_avg = all_dfs[0][first_cols].copy()

    for df_c in all_dfs:
        ld_col = [c for c in df_c.columns if c.startswith('LD_')][0]
        df_avg = df_avg.merge(df_c[['filename', ld_col]], on='filename', how='left')

    # Collect ld_cols from what's actually in df_avg -- avoids KeyError
    ld_cols = [c for c in df_avg.columns if c.startswith('LD_')]
    print(f"Averaging across {len(ld_cols)} conditions:")
    for col in ld_cols:
        print(f"  {col}")
    print()

    df_avg['mean_L_over_D'] = df_avg[ld_cols].mean(axis=1, skipna=True)
    df_avg['min_L_over_D']  = df_avg[ld_cols].min(axis=1, skipna=True)
    df_avg['n_valid']       = df_avg[ld_cols].notna().sum(axis=1)
    df_avg = df_avg.sort_values('mean_L_over_D', ascending=False, na_position='last')

    avg_file = OUTPUT_DIR / "lookup_table_averaged_all_conditions.csv"
    df_avg.to_csv(avg_file, index=False)
    print(f"Saved averaged ranking -> {avg_file.name}")

    best_avg = df_avg.dropna(subset=['mean_L_over_D']).iloc[0]
    latent_cols = [f'p{i}' for i in range(1, 7)]
    best_avg_dict = {
        'filename':           best_avg['filename'],
        'mean_L_over_D':      float(best_avg['mean_L_over_D']),
        'min_L_over_D':       float(best_avg['min_L_over_D']),
        'n_conditions_valid': int(best_avg['n_valid']),
        'latent': [float(best_avg[c]) for c in latent_cols if c in best_avg.index],
    }
    best_avg_file = OUTPUT_DIR / "best_baseline_foil_averaged.json"
    with open(best_avg_file, 'w') as f:
        json.dump(best_avg_dict, f, indent=2)
    print(f"Saved best all-around  -> {best_avg_file.name}")
    print()
    print("TOP 5 ALL-AROUND FOILS:")
    show_cols = ['filename', 'mean_L_over_D', 'min_L_over_D', 'n_valid']
    show_cols = [c for c in show_cols if c in df_avg.columns]
    print(df_avg[show_cols].head(5).to_string(index=False))
    print()
    print(f"Best foil: {best_avg_dict['filename']}  "
          f"mean L/D={best_avg_dict['mean_L_over_D']:.1f}  "
          f"min L/D={best_avg_dict['min_L_over_D']:.1f}")


if __name__ == "__main__":
    import sys

    if "--phase2" in sys.argv:
        # Re-run only Phase 2 averaging without the 2-hour Phase 1.
        run_phase2_averaging_only()

    elif "--rebaseline" in sys.argv:
        # ACTION ITEM (2/23): Re-pick best baseline JSONs with camber filter.
        # Use this when you want a low-camber starting point for NOM without
        # re-running the full 2-hour Phase 1 evaluation.
        #
        # Usage:
        #   python tools/build_lookup_table.py --rebaseline
        #   python tools/build_lookup_table.py --rebaseline --max-camber 0.03
        #
        # Default max_camber = 0.04 (4%c): blocks Eppler/highly cambered foils,
        # allows NACA 4-series and below. Adjust with --max-camber if needed.
        max_camber = 0.04
        if "--max-camber" in sys.argv:
            idx = sys.argv.index("--max-camber")
            try:
                max_camber = float(sys.argv[idx + 1])
            except (IndexError, ValueError):
                print("WARNING: --max-camber needs a value (e.g. --max-camber 0.03)")
                print("Using default: 0.04")
        rebaseline_with_camber_filter(max_camber=max_camber)

    else:
        build_lookup_table()