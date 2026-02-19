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
RE_LIST    = [150000, 440000]                   # slow speed and max speed from Ski Cat spreadsheet


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
        # Merge all 12 per-condition tables into one wide table.
        # Each row = one foil, each LD_ column = its L/D at one condition.
        df_avg = all_dfs[0][['filename','p1','p2','p3','p4','p5','p6']].copy()
        ld_cols = []
        for df_c in all_dfs:
            ld_col = [c for c in df_c.columns if c.startswith('LD_')][0]
            df_avg = df_avg.merge(
                df_c[['filename', ld_col]], on='filename', how='left'
            )
            ld_cols.append(ld_col)

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
    print("    nom_optimize(alpha=1.0, Re=440000)")
    print()
    print("  Use best all-around foil (averaged baseline):")
    print("    nom_optimize(alpha=1.0, Re=440000,")
    print("      lookup_baseline_path='outputs/best_baseline_foil_averaged.json')")
    print("=" * 60)


if __name__ == "__main__":
    build_lookup_table()