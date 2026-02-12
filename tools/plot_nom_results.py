# tools/plot_nom_result.py
import numpy as np
import matplotlib.pyplot as plt

def main(path="outputs/best_coords_nom.csv"):
    coords = np.loadtxt(path, delimiter=",", skiprows=1)
    x, y = coords[:, 0], coords[:, 1]

    plt.figure()
    plt.plot(x, y)
    plt.axis("equal")
    plt.grid(True)
    plt.title("Best NOM Airfoil Coordinates")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == "__main__":
    main()
