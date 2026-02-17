"""
check_dataset_thickness.py

---------------------------------------------------------------------------
WHAT THIS SCRIPT DOES:
---------------------------------------------------------------------------
Scans ALL airfoils in the training dataset and computes:
  - The minimum thickness (upper y - lower y) for each foil
  - The minimum thickness across the ENTIRE dataset

This gives us the value to use for min_thickness in geometry_penalty().
We should NOT just make up a number -- we derive it from the actual data.

WHY THIS MATTERS:
  If we set min_thickness too high, valid foils get penalized unfairly.
  If we set min_thickness too low, non-foil garbage shapes slip through.
  Using the actual dataset minimum ensures the constraint is physically
  grounded in the foils we actually trained on.

HOW TO RUN:
  python check_dataset_thickness.py

WHAT YOU'LL SEE:
  A table showing each foil's minimum thickness, plus a final line
  telling you the recommended min_thickness value to use.
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration -- update these paths to match your project structure
# ---------------------------------------------------------------------------
CSV_PATH       = "data/airfoil_latent_params.csv"   # path to your latent params CSV
AIRFOILS_DIR   = "airfoils_txt"                      # folder containing the .txt foil files
N_POINTS       = 40                                  # 40 points per surface (= 80 total)
INTERIOR_X_MIN = 0.05   # ignore the first 5% (near leading edge) when finding min thickness
INTERIOR_X_MAX = 0.95   # ignore the last 5%  (near trailing edge) when finding min thickness
# ---------------------------------------------------------------------------


def compute_min_thickness_for_foil(filepath: str, n_points: int = 40) -> float | None:
    """
    Load a single foil .txt file and compute its minimum thickness
    in the interior region [INTERIOR_X_MIN, INTERIOR_X_MAX].

    The .txt file has format:
      x  y
      (80 rows: first 40 = upper surface TE->LE, next 40 = lower surface LE->TE)

    Returns:
      float: minimum thickness in interior, or None if the file failed to load.
    """
    try:
        data = np.loadtxt(filepath, skiprows=1)  # skip the header row

        if data.shape != (2 * n_points, 2):
            return None  # unexpected format, skip this foil

        # Upper surface: rows 0..39, stored TE->LE (x goes 1 -> 0)
        upper_te_to_le = data[:n_points]
        # Reverse so upper goes LE->TE (x goes 0 -> 1) -- same direction as lower
        upper_01 = upper_te_to_le[::-1]

        # Lower surface: rows 40..79, stored LE->TE (x goes 0 -> 1) -- already correct
        lower_01 = data[n_points:2 * n_points]

        xu, yu = upper_01[:, 0], upper_01[:, 1]
        xl, yl = lower_01[:, 0], lower_01[:, 1]

        # Shared x grid for interpolation
        xg = np.linspace(0.0, 1.0, 120)

        # Interpolate both surfaces onto the shared grid
        yu_g = np.interp(xg, xu, yu)
        yl_g = np.interp(xg, xl, yl)

        # Thickness at each x location
        thickness = yu_g - yl_g

        # Apply interior mask (skip LE and TE regions)
        mask = (xg >= INTERIOR_X_MIN) & (xg <= INTERIOR_X_MAX)
        t_int = thickness[mask]

        if len(t_int) == 0:
            return None

        return float(np.min(t_int))

    except Exception:
        return None


def main():
    # Load the CSV to get the list of foil filenames
    df = pd.read_csv(CSV_PATH)

    if "filename" not in df.columns:
        raise ValueError(f"CSV at {CSV_PATH} must have a 'filename' column.")

    filenames = df["filename"].tolist()
    print(f"\nChecking thickness for {len(filenames)} foils in '{AIRFOILS_DIR}/'...\n")

    results = []   # list of (filename, min_thickness)
    failed  = []   # list of filenames that couldn't be loaded

    for fname in filenames:
        # Convert from .png or .dat to .txt if needed
        base = os.path.splitext(fname)[0]
        filepath = os.path.join(AIRFOILS_DIR, base + ".txt")

        if not os.path.exists(filepath):
            failed.append(fname)
            continue

        t_min = compute_min_thickness_for_foil(filepath, n_points=N_POINTS)

        if t_min is None:
            failed.append(fname)
        else:
            results.append((fname, t_min))

    if not results:
        print("ERROR: Could not compute thickness for any foil. Check file paths.")
        return

    # Sort by minimum thickness (smallest first) to find worst cases
    results.sort(key=lambda x: x[1])

    # Print the 10 thinnest foils
    print("=== 10 THINNEST FOILS IN DATASET ===")
    for fname, t_min in results[:10]:
        print(f"  {fname:<40s}  min_thickness = {t_min:.6f}")

    # Overall dataset minimum
    all_t_min = [t for _, t in results]
    dataset_min = min(all_t_min)
    dataset_mean = sum(all_t_min) / len(all_t_min)

    print(f"\n=== DATASET THICKNESS SUMMARY ===")
    print(f"  Foils successfully processed: {len(results)}")
    print(f"  Foils failed/skipped:         {len(failed)}")
    print(f"  Min thickness (dataset-wide): {dataset_min:.6f}")
    print(f"  Mean min thickness:           {dataset_mean:.6f}")
    print(f"\n=== RECOMMENDATION ===")
    print(f"  Use min_thickness = {max(0.01, dataset_min):.4f} in geometry_penalty()")
    print(f"  (This is the smallest interior thickness seen in any training foil)")
    print(f"  In nom_driver.py, set:  min_thickness = {max(0.01, dataset_min):.4f}")
    print()


if __name__ == "__main__":
    main()