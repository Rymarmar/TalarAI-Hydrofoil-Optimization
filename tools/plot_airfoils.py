import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

BASELINE_CSV = OUT / "baseline_coords.csv"
NOM_CSV = OUT / "best_coords_nom.csv"
RANDOM_CSV = OUT / "best_coords_random.csv"  # optional
NOM_SUMMARY = OUT / "nom_summary.json"


def load_coords(csv_path: Path) -> np.ndarray:
    """
    Expected CSV format: two columns (x,y) with or without headers.
    """
    df = pd.read_csv(csv_path)

    # If it has weird column names, just take first two cols.
    if df.shape[1] >= 2:
        xy = df.iloc[:, :2].to_numpy(dtype=float)
        return xy

    raise ValueError(f"Could not read x,y from {csv_path}")


def annotate_from_nom_summary(ax):
    if not NOM_SUMMARY.exists():
        ax.text(
            0.02, 0.02,
            "nom_summary.json not found (skipping metrics annotation)",
            transform=ax.transAxes
        )
        return

    s = json.loads(NOM_SUMMARY.read_text())
    alpha = s.get("alpha")
    Re = s.get("Re")
    cl = s.get("best_CL")
    cd = s.get("best_CD")
    total = s.get("best_total")
    pen = s.get("best_penalty")
    obj = s.get("best_objective")

    msg = (
        f"NOM best @ alpha={alpha}, Re={Re}\n"
        f"CL={cl:.4f}, CD={cd:.5f}\n"
        f"objective={obj:.6f}, penalty={pen:.6f}, total={total:.6f}"
    )
    ax.text(0.02, 0.98, msg, transform=ax.transAxes, va="top")


def main():
    if not BASELINE_CSV.exists():
        raise FileNotFoundError(f"Missing {BASELINE_CSV}. Run demo_random_search first to generate baseline.")

    if not NOM_CSV.exists():
        raise FileNotFoundError(f"Missing {NOM_CSV}. Run nom_driver first to generate NOM result.")

    baseline = load_coords(BASELINE_CSV)
    nom = load_coords(NOM_CSV)

    plt.figure()
    plt.plot(baseline[:, 0], baseline[:, 1], label="Baseline")
    plt.plot(nom[:, 0], nom[:, 1], label="NOM Optimized")

    # Optional: include random search result if it exists
    if RANDOM_CSV.exists():
        rs = load_coords(RANDOM_CSV)
        plt.plot(rs[:, 0], rs[:, 1], label="Random Search (optional)", linestyle="--")

    plt.title("Hydrofoil Cross-Section: Baseline vs NOM Optimized")
    plt.xlabel("x (chordwise)")
    plt.ylabel("y (thickness/camber)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    ax = plt.gca()
    annotate_from_nom_summary(ax)

    save_path = OUT / "airfoil_overlay.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot -> {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
