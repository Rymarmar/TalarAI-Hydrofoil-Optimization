"""
diagnose_seed_failure.py

Run this from your project root:
    python diagnose_seed_failure.py

This script simulates what find_valid_seed() does WITHOUT needing the
decoder or NeuralFoil. It loads real training foil .txt files directly
and runs them through geometry_penalty() to find out WHICH constraint
is rejecting every single foil.

This tells us definitively whether the problem is:
  A) geometry_penalty() is too strict (rejects valid foils)
  B) The coordinate convention (_split_upper_lower) is wrong
  C) The constraint thresholds are miscalibrated
  D) Something in how coords are assembled
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# CONFIGURE THESE PATHS to match your project
# ---------------------------------------------------------------------------
CSV_PATH     = "pipeline/airfoil_latent_params_6.csv"   # adjust if needed
AIRFOILS_DIR = "airfoils_txt"
N_POINTS     = 40
MAX_FOILS    = 1647   # how many to test (keep small for speed)
# ---------------------------------------------------------------------------

# ---- inline the geometry_penalty logic so we don't need the full package ----

def relu(x):
    return float(max(0.0, x))

def _split_upper_lower(coords, n_points=40):
    """Mirrors constraints.py: coords[:40]=upper TE->LE, coords[40:]=lower LE->TE"""
    c = np.asarray(coords, dtype=float)
    upper_te_to_le = c[:n_points]
    lower_le_to_te = c[n_points:2*n_points]
    upper_01 = upper_te_to_le[::-1].copy()   # flip to LE->TE
    lower_01 = lower_le_to_te.copy()
    return upper_01, lower_01

def _interp_profiles(upper_01, lower_01, n_bins=120):
    xu, yu = upper_01[:, 0], upper_01[:, 1]
    xl, yl = lower_01[:, 0], lower_01[:, 1]
    xg = np.linspace(0.0, 1.0, n_bins)
    yu_g = np.interp(xg, xu, yu)
    yl_g = np.interp(xg, xl, yl)
    return xg, yu_g, yl_g

def geometry_penalty_diagnose(coords,
                               n_points=40,
                               min_thickness=0.04,
                               max_thickness=0.14,
                               thickness_x_min=0.05,
                               thickness_x_max=0.90,
                               camber_max_abs=0.08,
                               te_gap_max=0.01,
                               le_gap_max=0.01,
                               max_abs_y=0.25,
                               x_tol=0.02):
    c = np.asarray(coords, dtype=float)
    if c.ndim != 2 or c.shape[1] != 2 or not np.all(np.isfinite(c)):
        return float("inf"), "coords_invalid"

    xmin = float(np.min(c[:, 0]))
    xmax = float(np.max(c[:, 0]))
    if xmin < (0.0 - x_tol) or xmax > (1.0 + x_tol):
        return float("inf"), f"x_out_of_range: xmin={xmin:.4f} xmax={xmax:.4f}"

    maxy = float(np.max(np.abs(c[:, 1])))
    if maxy > max_abs_y:
        return float("inf"), f"y_too_large: max|y|={maxy:.4f}"

    upper_01, lower_01 = _split_upper_lower(c, n_points=n_points)

    te_gap = float(np.linalg.norm(upper_01[-1] - lower_01[-1]))
    le_gap = float(np.linalg.norm(upper_01[0]  - lower_01[0]))

    xg, yu, yl = _interp_profiles(upper_01, lower_01)
    thickness = yu - yl
    camber    = 0.5 * (yu + yl)

    if np.any(thickness < -1e-4):
        min_t_full = float(np.min(thickness))
        x_cross    = float(xg[np.argmin(thickness)])
        return float("inf"), f"surface_crossing_FULL: min_t={min_t_full:.5f} at x={x_cross:.3f}"

    m_int = (xg >= thickness_x_min) & (xg <= thickness_x_max)
    t_int   = thickness[m_int]
    c_int   = camber[m_int]
    min_t   = float(np.min(t_int))
    max_t   = float(np.max(t_int))
    max_cam = float(np.max(np.abs(c_int)))

    if min_t < 0.0:
        return float("inf"), f"surface_crossing_INT: min_t={min_t:.5f}"

    if min_t < min_thickness:
        return float("inf"), f"too_thin: min_t={min_t:.5f} < {min_thickness}"

    if max_t > max_thickness:
        return float("inf"), f"too_thick: max_t={max_t:.5f} > {max_thickness}"

    if max_cam > camber_max_abs:
        return float("inf"), f"camber: max_cam={max_cam:.5f} > {camber_max_abs}"

    return 0.0, f"PASS (te_gap={te_gap:.4f} le_gap={le_gap:.4f} min_t={min_t:.4f} max_t={max_t:.4f} cam={max_cam:.4f})"


def load_foil_as_coords(filepath, n_points=40):
    """
    Load a Selig .txt file and assemble coords in talarai_pipeline.py convention:
      coords[:40] = upper TE->LE  (x: 1->0)
      coords[40:] = lower LE->TE  (x: 0->1)
    Selig .txt layout:
      rows 0-39  = lower surface TE->LE  (x: 1->0, y: negative)
      rows 40-79 = upper surface LE->TE  (x: 0->1, y: positive)
    """
    data = np.loadtxt(filepath, skiprows=1)
    if data.shape != (2 * n_points, 2):
        return None, f"bad shape {data.shape}"

    lower_te2le = data[:n_points]     # rows  0-39: lower TE->LE
    upper_le2te = data[n_points:]     # rows 40-79: upper LE->TE

    # Pipeline convention: upper TE->LE + lower LE->TE
    upper_te2le = upper_le2te[::-1]              # flip upper to TE->LE
    lower_le2te = lower_te2le[::-1]              # flip lower to LE->TE

    coords = np.vstack([upper_te2le, lower_le2te])
    return coords, "ok"


def main():
    print("=" * 70)
    print("SEED FAILURE DIAGNOSTIC")
    print("Simulating geometry_penalty() on real training foils")
    print("=" * 70)

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV not found at {CSV_PATH}")
        print("Edit CSV_PATH at the top of this script.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    filenames = df["filename"].tolist()

    print(f"\nLoaded {len(filenames)} foils from {CSV_PATH}")
    print(f"Testing first {MAX_FOILS} (edit MAX_FOILS to change)\n")

    reject_reasons = Counter()
    pass_count  = 0
    fail_count  = 0
    skip_count  = 0

    # Also track raw geometry stats for passing foils
    all_min_t = []
    all_max_t = []
    all_cam   = []

    for i, fname in enumerate(filenames[:MAX_FOILS]):
        base      = os.path.splitext(fname)[0]
        filepath  = os.path.join(AIRFOILS_DIR, base + ".txt")

        if not os.path.exists(filepath):
            skip_count += 1
            continue

        coords, load_status = load_foil_as_coords(filepath)
        if coords is None:
            skip_count += 1
            continue

        # ---- Run with CURRENT constraint values (matching nom_driver.py) ----
        pen, reason = geometry_penalty_diagnose(
            coords,
            min_thickness  = 0.04,
            max_thickness  = 0.14,
            camber_max_abs = 0.08,
            te_gap_max     = 0.01,
            le_gap_max     = 0.01,
        )

        # Also compute raw stats regardless of pass/fail
        upper_01, lower_01 = _split_upper_lower(coords)
        xg, yu, yl = _interp_profiles(upper_01, lower_01)
        thickness  = yu - yl
        camber     = 0.5 * (yu + yl)
        m_int      = (xg >= 0.05) & (xg <= 0.90)
        t_int      = thickness[m_int]
        c_int      = camber[m_int]

        # First word of reason is the category
        category = reason.split(":")[0].split("(")[0].strip()
        reject_reasons[category] += 1

        status = "PASS" if pen == 0.0 else "FAIL"
        if status == "PASS":
            pass_count += 1
            all_min_t.append(float(np.min(t_int)))
            all_max_t.append(float(np.max(t_int)))
            all_cam.append(float(np.max(np.abs(c_int))))
        else:
            fail_count += 1

        print(f"[{i:3d}] {base:<30s}  {status}  {reason[:80]}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Tested:  {MAX_FOILS}")
    print(f"  Skipped: {skip_count}  (file not found or bad shape)")
    print(f"  PASS:    {pass_count}")
    print(f"  FAIL:    {fail_count}")
    print()
    print("FAILURE REASONS (what constraint is rejecting foils):")
    for reason, count in reject_reasons.most_common():
        if reason != "PASS":
            print(f"  {reason:<40s}  {count} foils")

    if pass_count > 0:
        print(f"\nPASSING FOIL STATS:")
        print(f"  min_t range:  [{min(all_min_t):.4f}, {max(all_min_t):.4f}]")
        print(f"  max_t range:  [{min(all_max_t):.4f}, {max(all_max_t):.4f}]")
        print(f"  camber range: [{min(all_cam):.4f}, {max(all_cam):.4f}]")
    else:
        print("\nNO FOILS PASS. Checking what thresholds would be needed...")
        # Run again with very loose thresholds to see raw stats
        all_min_t_raw = []
        all_max_t_raw = []
        all_cam_raw   = []
        for fname in filenames[:MAX_FOILS]:
            base     = os.path.splitext(fname)[0]
            filepath = os.path.join(AIRFOILS_DIR, base + ".txt")
            if not os.path.exists(filepath):
                continue
            coords, _ = load_foil_as_coords(filepath)
            if coords is None:
                continue
            upper_01, lower_01 = _split_upper_lower(coords)
            xg, yu, yl = _interp_profiles(upper_01, lower_01)
            thickness  = yu - yl
            camber     = 0.5 * (yu + yl)
            m_int      = (xg >= 0.05) & (xg <= 0.90)
            if not np.any(m_int):
                continue
            t_int = thickness[m_int]
            c_int = camber[m_int]
            if np.any(thickness < -1e-4):
                continue   # true surface crossing -- skip
            all_min_t_raw.append(float(np.min(t_int)))
            all_max_t_raw.append(float(np.max(t_int)))
            all_cam_raw.append(float(np.max(np.abs(c_int))))

        if all_min_t_raw:
            print(f"\n  Raw interior min_thickness:  min={min(all_min_t_raw):.4f}  "
                  f"max={max(all_min_t_raw):.4f}  "
                  f"mean={sum(all_min_t_raw)/len(all_min_t_raw):.4f}")
            print(f"  Raw interior max_thickness:  min={min(all_max_t_raw):.4f}  "
                  f"max={max(all_max_t_raw):.4f}  "
                  f"mean={sum(all_max_t_raw)/len(all_max_t_raw):.4f}")
            print(f"  Raw interior max_camber:     min={min(all_cam_raw):.4f}  "
                  f"max={max(all_cam_raw):.4f}  "
                  f"mean={sum(all_cam_raw)/len(all_cam_raw):.4f}")
            print()
            print("  SUGGESTED THRESHOLDS (to let most foils pass):")
            print(f"    min_thickness  = {max(0.005, min(all_min_t_raw)*0.9):.4f}  "
                  f"(dataset min interior thickness * 0.9)")
            print(f"    max_thickness  = {max(all_max_t_raw)*1.1:.4f}  "
                  f"(dataset max interior thickness * 1.1)")
            print(f"    camber_max_abs = {max(all_cam_raw)*1.1:.4f}  "
                  f"(dataset max camber * 1.1)")


if __name__ == "__main__":
    main()