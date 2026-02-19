"""
tools/find_dataset_outliers.py

Find foils with extreme y-coordinate values that might be corrupting
the dataset statistics.
"""

from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AIRFOILS_DIR = PROJECT_ROOT / "airfoils_txt"

def find_outliers():
    """Find foils with extreme y values."""
    
    txt_files = list(AIRFOILS_DIR.glob("*.txt"))
    print(f"Scanning {len(txt_files)} files for outliers...\n")
    
    results = []
    
    for txt_path in txt_files:
        try:
            data = np.loadtxt(txt_path, skiprows=1)
            if data.ndim != 2 or data.shape[1] != 2:
                continue
            
            y_vals = data[:, 1]
            max_abs_y = float(np.max(np.abs(y_vals)))
            max_y = float(np.max(y_vals))
            min_y = float(np.min(y_vals))
            
            results.append({
                'file': txt_path.name,
                'max_abs_y': max_abs_y,
                'max_y': max_y,
                'min_y': min_y,
            })
        except:
            continue
    
    # Sort by max_abs_y (largest first)
    results.sort(key=lambda x: x['max_abs_y'], reverse=True)
    
    print("=" * 70)
    print("TOP 20 FOILS WITH LARGEST |Y| VALUES")
    print("=" * 70)
    print(f"{'Filename':<40} {'Max |y|':>10} {'Max y':>10} {'Min y':>10}")
    print("-" * 70)
    
    for r in results[:20]:
        print(f"{r['file']:<40} {r['max_abs_y']:>10.6f} {r['max_y']:>10.6f} {r['min_y']:>10.6f}")
    
    print()
    print("=" * 70)
    print("STATISTICS")
    print("=" * 70)
    
    all_max_abs = [r['max_abs_y'] for r in results]
    
    print(f"Largest max |y|:  {all_max_abs[0]:.6f}  ({results[0]['file']})")
    print(f"2nd largest:      {all_max_abs[1]:.6f}  ({results[1]['file']})")
    print(f"3rd largest:      {all_max_abs[2]:.6f}  ({results[2]['file']})")
    print()
    print(f"Median max |y|:   {np.median(all_max_abs):.6f}")
    print(f"Mean max |y|:     {np.mean(all_max_abs):.6f}")
    print(f"95th percentile:  {np.percentile(all_max_abs, 95):.6f}")
    print(f"99th percentile:  {np.percentile(all_max_abs, 99):.6f}")
    print()
    
    # Recommendation based on 99th percentile + 10% buffer
    p99 = np.percentile(all_max_abs, 99)
    recommended = p99 * 1.1
    
    print("=" * 70)
    print("RECOMMENDED VALUE FOR constraints.py")
    print("=" * 70)
    print()
    print(f"Using 99th percentile + 10% buffer (ignores top 1% outliers):")
    print(f"    if np.any(np.abs(y_all) > {recommended:.4f}):")
    print()
    print(f"This is much tighter than the 0.5500 recommended before,")
    print(f"which was based on the outlier at y={all_max_abs[0]:.2f}.")
    print()
    
    # Check if there's a big gap
    if all_max_abs[0] > 2 * all_max_abs[10]:
        print("⚠️  WARNING: Top foil is an OUTLIER (2x larger than #10).")
        print(f"   Consider inspecting: {results[0]['file']}")
    
    print("=" * 70)

if __name__ == "__main__":
    find_outliers()