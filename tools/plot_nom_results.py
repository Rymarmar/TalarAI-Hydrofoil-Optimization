# tools/plot_nom_results.py
import os
import numpy as np
import matplotlib.pyplot as plt


def _load_best_latent(outputs_dir: str = "outputs") -> np.ndarray | None:
    """
    Load best 6 latent params if available.

    Priority:
      1) outputs/best_latent_nom.csv (human readable)
      2) outputs/best_latent_nom.npy (numpy)

    Returns:
      array shape (6,) or None if not found.
    """
    csv_path = os.path.join(outputs_dir, "best_latent_nom.csv")
    npy_path = os.path.join(outputs_dir, "best_latent_nom.npy")

    if os.path.exists(csv_path):
        # CSV saved as 1 row with header
        z = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        z = np.asarray(z, dtype=float).reshape(-1)
        return z

    if os.path.exists(npy_path):
        z = np.load(npy_path)
        z = np.asarray(z, dtype=float).reshape(-1)
        return z

    return None


def main(coords_path: str = "outputs/best_coords_nom.csv", outputs_dir: str = "outputs", n_points: int = 40):
    """
    Plot the best NOM airfoil coordinates and annotate the figure with the optimized 6 params.

    Note (meeting):
      - coords are decoded geometry from the 6 optimized latent params
      - we plot upper and lower separately to avoid drawing a fake straight line across TE
    """
    coords = np.loadtxt(coords_path, delimiter=",", skiprows=1)
    coords = np.asarray(coords, dtype=float)

    # Split upper/lower using project convention
    # upper: first n_points points (TE->LE)
    # lower: last  n_points points (LE->TE)
    if coords.shape[0] == 2 * n_points:
        lower = coords[:n_points]
        upper = coords[n_points:]
    else:
        # fallback: just plot everything if shape is unexpected
        lower = coords
        upper = None

    # Load latent params for annotation
    z = _load_best_latent(outputs_dir=outputs_dir)

    plt.figure()
    plt.plot(upper[:, 0], upper[:, 1], label="upper (TE→LE)")
    if lower is not None:
        plt.plot(lower[:, 0], lower[:, 1], label="lower (LE→TE)")

    plt.axis("equal")
    plt.grid(True)
    plt.title("Best NOM Airfoil (decoded from 6 latent params)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")

    # Add latent params text box
    if z is not None and z.size == 6:
        text = "Best latent (p1..p6):\n" + "\n".join([f"p{i+1} = {z[i]: .6f}" for i in range(6)])
        plt.gcf().text(
            0.72, 0.25, text,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
        )

    plt.show()


if __name__ == "__main__":
    main()
