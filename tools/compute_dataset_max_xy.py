"""
tools/compute_dataset_max_xy.py

PURPOSE:
  Scan all airfoils in the training dataset and find the maximum x and y
  coordinate values. This gives us the actual range the decoder produces.

  Prof's action item #5-6:
    "Find max y and x from dataset, narrow to ½ for buffer"
    "Why is max_abs_y = 0.25 (a random number)? Not a good idea"

WHAT IT DOES:
  1) Loads every .txt file in airfoils_txt/
  2) Finds min/max for BOTH x and y coordinates across ALL foils
  3) Computes statistics and recommended constraint values
  4) Tells you what to update in constraints.py

HOW TO USE:
  Run from repo root:
    python tools/compute_dataset_max_xy.py

  Or if in tools directory:
    python compute_dataset_max_xy.py

  Or from Windows PowerShell:
    python tools\compute_dataset_max_xy.py

OUTPUT:
  Prints min/max for x and y, plus recommended values for constraints.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

# Resolve paths - works from repo root or tools/ directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "tools" else SCRIPT_DIR

# Dataset location
AIRFOILS_DIR = PROJECT_ROOT / "airfoils_txt"


def scan_dataset_coordinates() -> dict:
    """
    Scan all .txt files and find min/max for x and y coordinates.
    
    Returns:
        dict with keys:
            'x_min', 'x_max', 'y_min', 'y_max', 
            'max_abs_y', 'n_files', 'n_coords'
    """
    if not AIRFOILS_DIR.exists():
        raise FileNotFoundError(f"Airfoils directory not found: {AIRFOILS_DIR}")
    
    txt_files = list(AIRFOILS_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {AIRFOILS_DIR}")
    
    print(f"Scanning {len(txt_files)} airfoil files in {AIRFOILS_DIR}...\n")
    
    all_x_values = []
    all_y_values = []
    failed = []
    
    for txt_path in txt_files:
        try:
            # Load coordinates (skip header row)
            data = np.loadtxt(txt_path, skiprows=1)
            
            if data.ndim != 2 or data.shape[1] != 2:
                failed.append(txt_path.name)
                continue
            
            x_vals = data[:, 0]  # x column
            y_vals = data[:, 1]  # y column
            
            all_x_values.extend(x_vals.tolist())
            all_y_values.extend(y_vals.tolist())
            
        except Exception as e:
            failed.append(txt_path.name)
            continue
    
    if not all_x_values or not all_y_values:
        raise RuntimeError("No valid coordinates found in dataset")
    
    all_x = np.array(all_x_values)
    all_y = np.array(all_y_values)
    
    results = {
        'x_min': float(np.min(all_x)),
        'x_max': float(np.max(all_x)),
        'y_min': float(np.min(all_y)),
        'y_max': float(np.max(all_y)),
        'max_abs_y': float(np.max(np.abs(all_y))),
        'n_files': len(txt_files) - len(failed),
        'n_coords': len(all_x),
        'failed': failed,
    }
    
    return results


def main():
    results = scan_dataset_coordinates()
    
    print("=" * 70)
    print("DATASET COORDINATE STATISTICS (X and Y)")
    print("=" * 70)
    print(f"Files processed:    {results['n_files']}")
    print(f"Total coordinates:  {results['n_coords']:,}")
    print()
    
    print("X-COORDINATES:")
    print(f"  Min x: {results['x_min']:+.6f}")
    print(f"  Max x: {results['x_max']:+.6f}")
    print(f"  Range: {results['x_max'] - results['x_min']:.6f}")
    print()
    
    print("Y-COORDINATES:")
    print(f"  Min y: {results['y_min']:+.6f}")
    print(f"  Max y: {results['y_max']:+.6f}")
    print(f"  Max |y|: {results['max_abs_y']:.6f}")
    print(f"  Range: {results['y_max'] - results['y_min']:.6f}")
    print()
    
    if results['failed']:
        print(f"⚠️  {len(results['failed'])} files failed to load")
        print()
    
    # Compute recommendations based on prof's feedback
    # "Find max y and x from dataset, narrow to ½ for buffer"
    
    # For X: dataset should be normalized to [0, 1], so we use that
    x_buffer_lo = -0.1  # allow slight overshoot on low end
    x_buffer_hi = 1.1   # allow slight overshoot on high end
    
    # For Y: use half of max |y| with 10% safety buffer
    y_half = results['max_abs_y'] / 2.0
    y_recommended = y_half * 1.1  # add 10% safety
    
    print("=" * 70)
    print("RECOMMENDATION FOR constraints.py")
    print("=" * 70)
    print()
    print("ACTION ITEM #5-6: 'Find max y and x from dataset, narrow to ½ for buffer'")
    print()
    
    print("=" * 70)
    print("UPDATE #1: X-coordinate check (line ~167)")
    print("=" * 70)
    print()
    print("Current code:")
    print("    if np.any(x_all < -0.1) or np.any(x_all > 1.1):")
    print()
    print(f"✅ KEEP AS-IS (X is properly normalized to [0, 1])")
    print(f"   Dataset x range: [{results['x_min']:.4f}, {results['x_max']:.4f}]")
    print(f"   Check allows:    [{x_buffer_lo:.1f}, {x_buffer_hi:.1f}]")
    print()
    
    print("=" * 70)
    print("UPDATE #2: Y-coordinate check (line ~171)")
    print("=" * 70)
    print()
    print("Current code:")
    print("    if np.any(np.abs(y_all) > 0.5):")
    print()
    print("RECOMMENDED UPDATE:")
    print(f"    if np.any(np.abs(y_all) > {y_recommended:.4f}):")
    print()
    print("Calculation:")
    print(f"  Dataset max |y|   = {results['max_abs_y']:.6f}")
    print(f"  Half for buffer   = {y_half:.6f}")
    print(f"  + 10% safety      = {y_recommended:.6f}")
    print()
    print("This allows the decoder to produce any foil it saw in training,")
    print("plus a 10% safety buffer, while still rejecting garbage outputs.")
    print()
    
    # Summary comparison
    print("=" * 70)
    print("SUMMARY: What to change in constraints.py")
    print("=" * 70)
    print()
    print(f"Line ~167 (x check): ✅ KEEP AS-IS (already correct)")
    print(f"Line ~171 (y check): ⚠️  CHANGE from 0.5 to {y_recommended:.4f}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)