"""
tools/build_lookup_table.py

WHAT THIS FILE DOES:
  Evaluates all ~1647 training foils at every (alpha, Re) condition,
  picks the BEST foil at each condition (with geometry filtering),
  then averages across conditions to find the best all-around foil.

  The winner becomes the STARTING POINT for the NOM optimizer.

WHY GEOMETRY FILTERING MATTERS:
  Without it, the best-by-L/D foil might violate NOM's constraints
  (e.g. too cambered, too thin). NOM would then get penalty=1000
  every epoch and never improve. By filtering here, the baseline
  is guaranteed valid from the start.

OUTPUTS (what nom_driver.py actually needs):
  best_baseline_foil_alpha{a}_Re{Re}.json  -- per-condition best (REQUIRED)
  best_baseline_foil_averaged.json         -- best all-around foil

OUTPUTS (reference / debugging -- not loaded by nom_driver):
  lookup_table_alpha{a}_Re{Re}.csv         -- full rankings per condition
  lookup_table_averaged_all_conditions.csv  -- averaged rankings

HOW TO RUN:
  python tools/build_lookup_table.py              # full run (~20 min)
  python tools/build_lookup_table.py --phase2     # re-average only (skip Phase 1)

CHANGE LOG:
  [FIX 3/3/26] Phase 2 averaged baseline applies geometry filtering.
  [FIX 3/9/26] Phase 1 per-condition baselines NOW ALSO geometry filtered.
    Previously, per-condition best was picked by raw L/D with NO geometry
    checks. This caused e61.png (y too large) to be selected despite
    violating constraints, leading to penalty=1000 and 0 valid epochs.
  [FIX 3/9/26] Removed redundant top_100 CSV outputs.
  [FIX 3/9/26] Single run does Phase 1 + Phase 2 automatically.
"""

from __future__ import annotations
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import warnings, os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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

ALPHA_LIST = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
RE_LIST    = [50000, 100000, 150000, 250000, 350000, 450000]

# ---------------------------------------------------------------------------
# GEOMETRY FILTER DEFAULTS
# These MUST match the defaults in nom_driver.py / constraints.py so that
# the baseline selected here is guaranteed to pass NOM's constraint checks.
# If you change these in nom_driver.py, update them here too.
# ---------------------------------------------------------------------------
DEFAULT_MAX_CAMBER        = 0.08   # 8%c — same as nom_driver.py default
DEFAULT_MIN_MAX_THICKNESS = 0.04   # same as nom_driver.py default
DEFAULT_MIN_THICKNESS     = 0.006  # same as nom_driver.py default
DEFAULT_MAX_THICKNESS     = 0.157  # same as nom_driver.py default


# ============================================================================
# SINGLE-CONDITION EVALUATION
# ============================================================================

def evaluate_condition(pipeline, all_latents, filenames, alpha, Re):
    """
    Evaluate all foils at one (alpha, Re) pair.
    Returns a DataFrame sorted best-first by L/D.
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
# GEOMETRY FILTER UTILITIES
# ============================================================================

def compute_max_camber(coords: np.ndarray,
                       x_min: float = 0.05,
                       x_max: float = 0.90) -> float:
    """
    Compute the maximum absolute camber of a foil from its (80,2) coords.
    Returns 999.0 if coords are bad.
    """
    if coords is None or coords.shape != (80, 2):
        return 999.0

    upper_te2le = coords[:40]
    lower_le2te = coords[40:]
    upper_le2te = upper_te2le[::-1]

    xu, yu = upper_le2te[:, 0], upper_le2te[:, 1]
    xl, yl = lower_le2te[:, 0], lower_le2te[:, 1]

    camber_line = (yu + yl) / 2.0

    mask = (xu >= x_min) & (xu <= x_max)
    if not np.any(mask):
        return 999.0

    return float(np.max(np.abs(camber_line[mask])))


def compute_thickness_range(coords: np.ndarray,
                            x_min: float = 0.05,
                            x_max: float = 0.90) -> tuple[float, float]:
    """
    Compute min and max thickness in the interior [x_min, x_max].
    Returns (0.0, 0.0) if coords are bad.
    """
    if coords is None or coords.shape != (80, 2):
        return 0.0, 0.0

    upper_te2le = coords[:40]
    lower_le2te = coords[40:]
    upper_le2te = upper_te2le[::-1]

    xu, yu = upper_le2te[:, 0], upper_le2te[:, 1]
    xl, yl = lower_le2te[:, 0], lower_le2te[:, 1]

    thickness = yu - yl

    mask = (xu >= x_min) & (xu <= x_max)
    if not np.any(mask):
        return 0.0, 0.0

    t_interior = thickness[mask]
    return float(np.min(t_interior)), float(np.max(t_interior))


def check_foil_geometry(coords: np.ndarray,
                        max_camber: float = DEFAULT_MAX_CAMBER,
                        min_max_thickness: float = DEFAULT_MIN_MAX_THICKNESS,
                        min_thickness: float = DEFAULT_MIN_THICKNESS,
                        max_thickness: float = DEFAULT_MAX_THICKNESS,
                        ) -> tuple[bool, dict]:
    """
    Check whether a foil passes the geometry constraints NOM enforces.
    Returns (passes, info_dict).
    """
    camber = compute_max_camber(coords)
    t_min, t_max = compute_thickness_range(coords)

    info = {
        "max_camber": camber,
        "t_min": t_min,
        "t_max": t_max,
    }

    if camber > max_camber:
        info["reject_reason"] = f"camber {camber*100:.1f}%c > {max_camber*100:.0f}%c limit"
        return False, info

    if t_max < min_max_thickness:
        info["reject_reason"] = f"peak thickness {t_max*100:.1f}%c < {min_max_thickness*100:.1f}%c minimum"
        return False, info

    if t_min < min_thickness:
        info["reject_reason"] = f"min thickness {t_min*100:.2f}%c < {min_thickness*100:.2f}%c minimum"
        return False, info

    if t_max > max_thickness:
        info["reject_reason"] = f"max thickness {t_max*100:.1f}%c > {max_thickness*100:.1f}%c limit"
        return False, info

    # Check y-range (matches constraints.py line 188)
    y_all = coords[:, 1]
    if np.any(np.abs(y_all) > 0.1964):
        info["reject_reason"] = f"y out of range (max |y|={np.max(np.abs(y_all)):.4f} > 0.1964)"
        return False, info

    info["reject_reason"] = None
    return True, info


# ============================================================================
# MAIN BUILD FUNCTION
# ============================================================================

def build_lookup_table():
    """
    Full pipeline: Phase 1 (evaluate all foils) + Phase 2 (average + pick best).
    One command, one run, everything done.
    """

    if not LATENT_CSV.exists():
        raise FileNotFoundError(f"Latent CSV not found: {LATENT_CSV}")
    df_csv = pd.read_csv(LATENT_CSV)
    latent_cols = [c for c in df_csv.columns if c.lower().startswith('p')]
    all_latents = df_csv[latent_cols].values.astype(np.float32)
    filenames   = df_csv['filename'].tolist() if 'filename' in df_csv.columns \
                  else [f"foil_{i}" for i in range(len(df_csv))]
    print(f"Loaded {len(all_latents)} foils from {LATENT_CSV.name}")

    print("Loading TalarAI pipeline (decoder + NeuralFoil)...")
    pipeline = TalarAIPipeline()
    print(f"Pipeline ready: {pipeline.decoder_path.name}")
    print(f"Running {len(ALPHA_LIST)} x {len(RE_LIST)} = "
          f"{len(ALPHA_LIST)*len(RE_LIST)} conditions...\n")

    # ------------------------------------------------------------------
    # PRE-COMPUTE GEOMETRY VALIDITY FOR ALL FOILS (done once, used everywhere)
    # [FIX 3/9/26] This used to only happen in Phase 2. Now it happens
    # upfront so Phase 1 can also filter per-condition baselines.
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PRE-COMPUTING GEOMETRY VALIDITY")
    print(f"  Constraints: max_camber={DEFAULT_MAX_CAMBER*100:.0f}%c  "
          f"min_max_t={DEFAULT_MIN_MAX_THICKNESS*100:.1f}%c  "
          f"min_t={DEFAULT_MIN_THICKNESS*100:.2f}%c  "
          f"max_t={DEFAULT_MAX_THICKNESS*100:.1f}%c")
    print("=" * 60)

    geom_valid_map = {}   # filename -> True/False
    geom_info_map  = {}   # filename -> info dict
    for i, (latent, fname) in enumerate(zip(all_latents, filenames)):
        if (i + 1) % 200 == 0 or i == 0:
            print(f"  [{i+1:4d}/{len(all_latents)}] checking geometry...")
        try:
            coords = pipeline.latent_to_coordinates(latent)
            passes, info = check_foil_geometry(coords)
            geom_valid_map[fname] = passes
            geom_info_map[fname]  = info
        except Exception:
            geom_valid_map[fname] = False
            geom_info_map[fname]  = {"reject_reason": "exception"}

    n_geom_pass = sum(1 for v in geom_valid_map.values() if v)
    print(f"\nFoils passing geometry filter: {n_geom_pass}/{len(all_latents)} "
          f"({100*n_geom_pass/len(all_latents):.1f}%)")
    if n_geom_pass == 0:
        print("ERROR: Zero foils pass geometry! Check your decoder weights "
              "and constraint settings.")
        return
    print()

    # ------------------------------------------------------------------
    # PHASE 1: Evaluate all foils at each (alpha, Re) condition
    # [FIX 3/9/26] Per-condition best now uses geometry filtering.
    # ------------------------------------------------------------------
    total_conditions = len(ALPHA_LIST) * len(RE_LIST)
    done = 0
    summary_rows = []

    for alpha in ALPHA_LIST:
        for Re in RE_LIST:
            done += 1
            tag = f"alpha{alpha:.1f}_Re{Re:.1e}"
            print("=" * 60)
            print(f"[{done}/{total_conditions}]  alpha={alpha}deg  Re={Re:.0e}")
            print("=" * 60)

            df = evaluate_condition(pipeline, all_latents, filenames, alpha, Re)

            # Save full ranking (reference only -- not loaded by nom_driver)
            lookup_file = OUTPUT_DIR / f"lookup_table_{tag}.csv"
            df.to_csv(lookup_file, index=False)
            print(f"  Saved full table  -> {lookup_file.name}")

            # [FIX 3/9/26] Pick best foil WITH geometry filtering.
            # Previously this was raw L/D, which could pick geometry-violating
            # foils like e61 (y too large) causing NOM to skip all epochs.
            df_valid = df.dropna(subset=['L_over_D'])

            # Apply geometry filter
            df_geom_ok = df_valid[
                df_valid['filename'].map(geom_valid_map).fillna(False)
            ]

            if df_geom_ok.empty:
                print(f"  WARNING: No geometry-valid foils at {tag}!")
                print(f"  Falling back to unfiltered best (NOM may struggle).")
                df_pick = df_valid
                geom_filtered = False
            else:
                df_pick = df_geom_ok
                geom_filtered = True

            if df_pick.empty:
                print("  WARNING: no valid foils at this condition, skipping JSON")
                continue

            best = df_pick.iloc[0]

            # Show if geometry filter changed the pick
            if geom_filtered and not df_valid.empty:
                raw_best = df_valid.iloc[0]
                if raw_best['filename'] != best['filename']:
                    raw_info = geom_info_map.get(raw_best['filename'], {})
                    print(f"  GEOMETRY FILTER CHANGED PICK:")
                    print(f"    Raw best:      {raw_best['filename']}  "
                          f"L/D={raw_best['L_over_D']:.1f}  "
                          f"reject={raw_info.get('reject_reason', '?')}")
                    print(f"    Filtered best: {best['filename']}  "
                          f"L/D={best['L_over_D']:.1f}")

            # Save the per-condition best JSON (this is what nom_driver loads)
            best_info = geom_info_map.get(best['filename'], {})
            best_dict = {
                'filename':          best['filename'],
                'alpha':             alpha,
                'Re':                Re,
                'CL':                float(best['CL']),
                'CD':                float(best['CD']),
                'L_over_D':          float(best['L_over_D']),
                'CD_over_CL':        float(best['CD_over_CL']),
                'geometry_filtered': geom_filtered,
                'max_camber':        float(best_info.get('max_camber', 0)),
                'peak_thickness':    float(best_info.get('t_max', 0)),
                'latent': [float(best[f'p{i}']) for i in range(1, 7)],
            }
            best_file = OUTPUT_DIR / f"best_baseline_foil_{tag}.json"
            with open(best_file, 'w') as f:
                json.dump(best_dict, f, indent=2)
            print(f"  Saved best foil   -> {best_file.name}")
            print(f"  Best: {best['filename']}  "
                  f"L/D={best['L_over_D']:.1f}  "
                  f"CL={best['CL']:.4f}  CD={best['CD']:.6f}  "
                  f"camber={best_info.get('max_camber', 0)*100:.1f}%c")

            summary_rows.append({
                'alpha':     alpha,
                'Re':        Re,
                'best_foil': best['filename'],
                'L_over_D':  round(float(best['L_over_D']), 1),
                'CL':        round(float(best['CL']), 4),
                'CD':        round(float(best['CD']), 6),
                'camber_pct': round(float(best_info.get('max_camber', 0)) * 100, 1),
                'geom_filtered': geom_filtered,
            })
            print()

    # ------------------------------------------------------------------
    # PHASE 2: Average L/D across all conditions, pick best all-around
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 2: AVERAGING L/D ACROSS ALL CONDITIONS")
    print("=" * 60)

    all_dfs = []
    for alpha in ALPHA_LIST:
        for Re in RE_LIST:
            tag = f"alpha{alpha:.1f}_Re{Re:.1e}"
            fpath = OUTPUT_DIR / f"lookup_table_{tag}.csv"
            if fpath.exists():
                df_c = pd.read_csv(fpath)[
                    ['filename', 'L_over_D', 'p1','p2','p3','p4','p5','p6']
                ]
                df_c = df_c.rename(columns={'L_over_D': f'LD_{tag}'})
                all_dfs.append(df_c)

    if all_dfs:
        df_avg = all_dfs[0][['filename','p1','p2','p3','p4','p5','p6']].copy()
        for df_c in all_dfs:
            ld_col = [c for c in df_c.columns if c.startswith('LD_')][0]
            df_avg = df_avg.merge(
                df_c[['filename', ld_col]], on='filename', how='left'
            )

        ld_cols = [c for c in df_avg.columns if c.startswith('LD_')]
        print(f"  Averaging across {len(ld_cols)} conditions")

        df_avg['mean_L_over_D'] = df_avg[ld_cols].mean(axis=1, skipna=True)
        df_avg['min_L_over_D']  = df_avg[ld_cols].min(axis=1, skipna=True)
        df_avg['n_valid']       = df_avg[ld_cols].notna().sum(axis=1)

        # Add geometry columns
        df_avg['geom_valid'] = df_avg['filename'].map(geom_valid_map).fillna(False)
        df_avg['max_camber'] = df_avg['filename'].map(
            lambda f: geom_info_map.get(f, {}).get('max_camber', 999.0)
        )
        df_avg['peak_thickness'] = df_avg['filename'].map(
            lambda f: geom_info_map.get(f, {}).get('t_max', 0.0)
        )

        df_avg = df_avg.sort_values(
            'mean_L_over_D', ascending=False, na_position='last'
        )

        # Save averaged ranking (reference only)
        avg_file = OUTPUT_DIR / "lookup_table_averaged_all_conditions.csv"
        df_avg.to_csv(avg_file, index=False)
        print(f"  Saved averaged ranking -> {avg_file.name}")

        # Pick best with geometry filter
        df_geom_valid = df_avg[df_avg['geom_valid'] == True].copy()
        df_geom_valid = df_geom_valid.dropna(subset=['mean_L_over_D'])

        if df_geom_valid.empty:
            print("  WARNING: No foils pass geometry filter! Using unfiltered best.")
            best_avg = df_avg.dropna(subset=['mean_L_over_D']).iloc[0]
            geom_filtered = False
        else:
            best_avg = df_geom_valid.iloc[0]
            geom_filtered = True
            unfiltered_best = df_avg.dropna(subset=['mean_L_over_D']).iloc[0]
            if unfiltered_best['filename'] != best_avg['filename']:
                print(f"\n  GEOMETRY FILTER CHANGED THE BASELINE:")
                print(f"    Without filter: {unfiltered_best['filename']}  "
                      f"mean L/D={unfiltered_best['mean_L_over_D']:.1f}  "
                      f"camber={unfiltered_best['max_camber']*100:.1f}%c")
                print(f"    With filter:    {best_avg['filename']}  "
                      f"mean L/D={best_avg['mean_L_over_D']:.1f}  "
                      f"camber={best_avg['max_camber']*100:.1f}%c")

        best_avg_dict = {
            'filename':           best_avg['filename'],
            'mean_L_over_D':      float(best_avg['mean_L_over_D']),
            'min_L_over_D':       float(best_avg['min_L_over_D']),
            'n_conditions_valid': int(best_avg['n_valid']),
            'geometry_filtered':  geom_filtered,
            'max_camber_actual':  float(best_avg.get('max_camber', 999.0)),
            'peak_thickness':     float(best_avg.get('peak_thickness', 0.0)),
            'geometry_limits': {
                'max_camber':        DEFAULT_MAX_CAMBER,
                'min_max_thickness': DEFAULT_MIN_MAX_THICKNESS,
                'min_thickness':     DEFAULT_MIN_THICKNESS,
                'max_thickness':     DEFAULT_MAX_THICKNESS,
            },
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

        print("  TOP 5 ALL-AROUND FOILS (geometry-filtered):")
        if not df_geom_valid.empty:
            show_cols = ['filename', 'mean_L_over_D', 'min_L_over_D',
                         'max_camber', 'peak_thickness', 'n_valid']
            print(df_geom_valid[show_cols].head(5).to_string(index=False))
        else:
            print("  (none pass geometry filter)")
    else:
        print("  WARNING: No condition CSVs found for averaging.")

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("BUILD COMPLETE — PER-CONDITION SUMMARY")
    print("=" * 60)
    if summary_rows:
        print(pd.DataFrame(summary_rows).to_string(index=False))
    print()
    print("READY TO OPTIMIZE. Run:")
    print(f"  python optimization/nom_driver.py")
    print("=" * 60)


# ============================================================================
# PHASE 2 ONLY (shortcut to skip the expensive Phase 1)
# ============================================================================

def run_phase2_averaging_only():
    """
    Re-run ONLY Phase 2 (averaging + geometry filtering) using existing
    per-condition CSVs from a previous Phase 1 run.

    Use this when you change geometry constraints but don't want to
    re-evaluate all 1647 foils at all conditions (~20 min).

    Usage:
        python tools/build_lookup_table.py --phase2
    """
    print("=" * 60)
    print("PHASE 2 ONLY: RE-AVERAGING WITH GEOMETRY FILTER")
    print(f"  max_camber={DEFAULT_MAX_CAMBER*100:.0f}%c  "
          f"min_max_thickness={DEFAULT_MIN_MAX_THICKNESS*100:.1f}%c")
    print("(Skipping Phase 1 — using existing lookup CSVs)")
    print("=" * 60)

    # Load pipeline for geometry checks
    print("Loading TalarAI pipeline...")
    pipeline = TalarAIPipeline()
    print(f"Pipeline ready: {pipeline.decoder_path.name}")
    print()

    # Load latent params
    df_csv = pd.read_csv(LATENT_CSV)
    latent_cols = [c for c in df_csv.columns if c.lower().startswith('p')]
    all_latents = df_csv[latent_cols].values.astype(np.float32)
    filenames = df_csv['filename'].tolist() if 'filename' in df_csv.columns \
                else [f"foil_{i}" for i in range(len(df_csv))]

    # Pre-compute geometry validity
    print(f"Checking geometry for {len(all_latents)} foils...")
    geom_valid_map = {}
    geom_info_map  = {}
    for i, (latent, fname) in enumerate(zip(all_latents, filenames)):
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(all_latents)}] checking geometry...")
        try:
            coords = pipeline.latent_to_coordinates(latent)
            passes, info = check_foil_geometry(coords)
            geom_valid_map[fname] = passes
            geom_info_map[fname]  = info
        except Exception:
            geom_valid_map[fname] = False
            geom_info_map[fname]  = {"reject_reason": "exception"}

    n_pass = sum(1 for v in geom_valid_map.values() if v)
    print(f"Foils passing geometry filter: {n_pass}/{len(all_latents)} "
          f"({100*n_pass/len(all_latents):.1f}%)")
    print()

    # Find existing per-condition CSVs
    existing_csvs = sorted(OUTPUT_DIR.glob("lookup_table_alpha*.csv"))
    existing_csvs = [f for f in existing_csvs if "averaged" not in f.name]

    if not existing_csvs:
        print(f"ERROR: No lookup_table_alpha*.csv files found in {OUTPUT_DIR}")
        print("Run the full build first:  python tools/build_lookup_table.py")
        return

    print(f"Found {len(existing_csvs)} condition CSV(s)")

    # --- Re-pick per-condition baselines with geometry filter ---
    print()
    print("Re-picking per-condition baselines with geometry filter...")
    for fpath in existing_csvs:
        tag = fpath.stem.replace("lookup_table_", "")
        df = pd.read_csv(fpath)

        if 'filename' not in df.columns or 'L_over_D' not in df.columns:
            continue

        df_valid = df.dropna(subset=['L_over_D'])
        df_geom_ok = df_valid[
            df_valid['filename'].map(geom_valid_map).fillna(False)
        ]

        df_pick = df_geom_ok if not df_geom_ok.empty else df_valid
        if df_pick.empty:
            continue

        best = df_pick.iloc[0]

        # Parse alpha and Re from tag
        try:
            parts   = tag.split('_Re')
            alpha_v = float(parts[0].replace('alpha', ''))
            Re_v    = float(parts[1])
        except Exception:
            alpha_v = Re_v = 0.0

        best_info = geom_info_map.get(best['filename'], {})
        best_dict = {
            'filename':          best['filename'],
            'alpha':             alpha_v,
            'Re':                Re_v,
            'CL':                float(best['CL']),
            'CD':                float(best['CD']),
            'L_over_D':          float(best['L_over_D']),
            'CD_over_CL':        float(best['CD_over_CL']),
            'geometry_filtered': not df_geom_ok.empty,
            'max_camber':        float(best_info.get('max_camber', 0)),
            'peak_thickness':    float(best_info.get('t_max', 0)),
            'latent': [float(best[f'p{i}']) for i in range(1, 7)],
        }
        best_file = OUTPUT_DIR / f"best_baseline_foil_{tag}.json"
        with open(best_file, 'w') as f:
            json.dump(best_dict, f, indent=2)
        print(f"  {tag}: {best['filename']}  L/D={best['L_over_D']:.1f}")

    # --- Average across conditions ---
    print()
    print("Averaging across all conditions...")
    all_dfs = []
    for fpath in existing_csvs:
        tag = fpath.stem.replace("lookup_table_", "")
        df_c = pd.read_csv(fpath)
        if 'L_over_D' not in df_c.columns or 'filename' not in df_c.columns:
            continue
        needed = ['filename', 'L_over_D'] + [f'p{i}' for i in range(1, 7)]
        df_c = df_c[[c for c in needed if c in df_c.columns]]
        df_c = df_c.rename(columns={'L_over_D': f'LD_{tag}'})
        all_dfs.append(df_c)

    if not all_dfs:
        print("ERROR: No valid CSVs could be loaded.")
        return

    first_cols = ['filename'] + [f'p{i}' for i in range(1, 7)]
    first_cols = [c for c in first_cols if c in all_dfs[0].columns]
    df_avg = all_dfs[0][first_cols].copy()

    for df_c in all_dfs:
        ld_col = [c for c in df_c.columns if c.startswith('LD_')][0]
        df_avg = df_avg.merge(df_c[['filename', ld_col]], on='filename', how='left')

    ld_cols = [c for c in df_avg.columns if c.startswith('LD_')]
    print(f"  Averaging across {len(ld_cols)} conditions")

    df_avg['mean_L_over_D'] = df_avg[ld_cols].mean(axis=1, skipna=True)
    df_avg['min_L_over_D']  = df_avg[ld_cols].min(axis=1, skipna=True)
    df_avg['n_valid']       = df_avg[ld_cols].notna().sum(axis=1)
    df_avg['geom_valid']    = df_avg['filename'].map(geom_valid_map).fillna(False)
    df_avg['max_camber']    = df_avg['filename'].map(
        lambda f: geom_info_map.get(f, {}).get('max_camber', 999.0)
    )
    df_avg['peak_thickness'] = df_avg['filename'].map(
        lambda f: geom_info_map.get(f, {}).get('t_max', 0.0)
    )

    df_avg = df_avg.sort_values('mean_L_over_D', ascending=False, na_position='last')

    avg_file = OUTPUT_DIR / "lookup_table_averaged_all_conditions.csv"
    df_avg.to_csv(avg_file, index=False)
    print(f"  Saved -> {avg_file.name}")

    # Pick best with geometry filter
    df_geom_valid = df_avg[df_avg['geom_valid'] == True].copy()
    df_geom_valid = df_geom_valid.dropna(subset=['mean_L_over_D'])

    if df_geom_valid.empty:
        print("  WARNING: No foils pass geometry filter! Using unfiltered best.")
        best_avg = df_avg.dropna(subset=['mean_L_over_D']).iloc[0]
        geom_filtered = False
    else:
        best_avg = df_geom_valid.iloc[0]
        geom_filtered = True

    latent_cols = [f'p{i}' for i in range(1, 7)]
    best_avg_dict = {
        'filename':           best_avg['filename'],
        'mean_L_over_D':      float(best_avg['mean_L_over_D']),
        'min_L_over_D':       float(best_avg['min_L_over_D']),
        'n_conditions_valid': int(best_avg['n_valid']),
        'geometry_filtered':  geom_filtered,
        'max_camber_actual':  float(best_avg.get('max_camber', 999.0)),
        'peak_thickness':     float(best_avg.get('peak_thickness', 0.0)),
        'geometry_limits': {
            'max_camber':        DEFAULT_MAX_CAMBER,
            'min_max_thickness': DEFAULT_MIN_MAX_THICKNESS,
            'min_thickness':     DEFAULT_MIN_THICKNESS,
            'max_thickness':     DEFAULT_MAX_THICKNESS,
        },
        'latent': [float(best_avg[c]) for c in latent_cols if c in best_avg.index],
    }
    best_avg_file = OUTPUT_DIR / "best_baseline_foil_averaged.json"
    with open(best_avg_file, 'w') as f:
        json.dump(best_avg_dict, f, indent=2)
    print(f"  Saved -> {best_avg_file.name}")
    print()

    print("TOP 5 ALL-AROUND FOILS (geometry-filtered):")
    if not df_geom_valid.empty:
        show_cols = ['filename', 'mean_L_over_D', 'min_L_over_D',
                     'max_camber', 'peak_thickness', 'n_valid']
        show_cols = [c for c in show_cols if c in df_geom_valid.columns]
        print(df_geom_valid[show_cols].head(5).to_string(index=False))
    else:
        print("  (none pass geometry filter)")

    print()
    print(f"Best foil: {best_avg_dict['filename']}  "
          f"mean L/D={best_avg_dict['mean_L_over_D']:.1f}")
    print()
    print("READY TO OPTIMIZE. Run:")
    print(f"  python optimization/nom_driver.py")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys

    if "--phase2" in sys.argv:
        run_phase2_averaging_only()
    else:
        build_lookup_table()