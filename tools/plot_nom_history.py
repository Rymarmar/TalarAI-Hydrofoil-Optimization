import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "outputs" / "nom_history.json"


def main():
    if not HIST.exists():
        raise FileNotFoundError(f"Missing {HIST}. Run: python -m optimization.nom_driver")

    data = json.loads(HIST.read_text())

    # Expect a list of dicts: [{"iter": i, "objective":..., "penalty":..., "total":...}, ...]
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("nom_history.json format unexpected (expected a non-empty list).")

    iters = np.array([row.get("iter", idx) for idx, row in enumerate(data)], dtype=int)
    obj = np.array([row.get("objective", np.nan) for row in data], dtype=float)
    pen = np.array([row.get("penalty", np.nan) for row in data], dtype=float)
    total = np.array([row.get("total", np.nan) for row in data], dtype=float)

    plt.figure()
    plt.plot(iters, total, label="Total (objective + penalty)")
    plt.plot(iters, obj, label="Objective")
    plt.plot(iters, pen, label="Penalty")

    plt.title("NOM Optimization Progress")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()

    save_path = ROOT / "outputs" / "nom_history_plot.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot -> {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
