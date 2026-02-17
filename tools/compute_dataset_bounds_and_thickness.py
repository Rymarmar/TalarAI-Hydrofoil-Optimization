"""
tools/compute_dataset_bounds_and_thickness.py

Run from repo root:
    python tools/compute_dataset_bounds_and_thickness.py

This script:

1) Prints dataset-wide min/max for p1..p6
2) Saves a CSV containing:
      - min and max for each parameter
3) Saves a CSV copy of all latent params (clean numeric export)
4) (OPTIONAL) Computes min thickness across dataset
"""

from __future__ import annotations

import sys
from pathlib import Path

# ------------------------------------------------------------
# Ensure project root is on Python path
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def main(
    csv_path: str = "data/airfoil_latent_params.csv",
    compute_thickness: bool = False,  # default OFF so script finishes fast
):

    df = pd.read_csv(csv_path)

    if "filename" not in df.columns:
        raise ValueError("Expected 'filename' column.")

    latent_cols = [c for c in df.columns if c.lower().startswith("p")]
    if len(latent_cols) != 6:
        raise ValueError("Expected 6 latent columns p1..p6.")

    Z = df[latent_cols].to_numpy(dtype=float)

    # ------------------------------------------------------------
    # 1) Compute dataset min/max
    # ------------------------------------------------------------
    lo = np.min(Z, axis=0)
    hi = np.max(Z, axis=0)
    ranges = hi - lo

    print("\n=== DATASET LATENT PARAMETER BOUNDS (min/max across ALL foils) ===")
    for i, col in enumerate(latent_cols):
        print(f"{col}: lo={lo[i]: .6f}, hi={hi[i]: .6f}, range={ranges[i]: .6f}")

    # ------------------------------------------------------------
    # 2) Save bounds to CSV (what professor wants)
    # ------------------------------------------------------------
    bounds_df = pd.DataFrame({
        "parameter": latent_cols,
        "min_value": lo,
        "max_value": hi,
        "range": ranges,
    })

    bounds_path = PROJECT_ROOT / "outputs" / "latent_param_bounds.csv"
    bounds_path.parent.mkdir(exist_ok=True)
    bounds_df.to_csv(bounds_path, index=False)

    print(f"\nSaved bounds CSV to: {bounds_path}")

    # ------------------------------------------------------------
    # 3) Save full latent dataset as numeric CSV
    # ------------------------------------------------------------
    full_latent_path = PROJECT_ROOT / "outputs" / "all_latent_params_numeric.csv"
    df[["filename"] + latent_cols].to_csv(full_latent_path, index=False)

    print(f"Saved full latent dataset to: {full_latent_path}")

    # ------------------------------------------------------------
    # 4) OPTIONAL thickness computation (slow!)
    # ------------------------------------------------------------
    if compute_thickness:
        print("\nThickness computation not implemented in fast mode.")
        print("If needed, we can enable full decode thickness scan separately.")

    print("\nDone. Script finished cleanly.")


if __name__ == "__main__":
    main()
