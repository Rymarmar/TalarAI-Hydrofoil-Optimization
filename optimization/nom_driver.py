import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.talarai_pipeline import TalarAIPipeline
from optimization.objective import default_objective
from optimization.constraints import make_default_latent_bounds, total_penalty


def load_latent_dataset(csv_path: str = "data/airfoil_latent_params.csv") -> np.ndarray:
    df = pd.read_csv(csv_path)
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] != 6:
        raise ValueError(
            f"Expected 6 numeric latent columns, found {numeric_df.shape[1]} in {csv_path}"
        )
    return numeric_df.values.astype(float)


def propose_latent(mu, sigma, k: float = 2.0) -> np.ndarray:
    """
    Propose a candidate latent vector from a clipped normal distribution.
    This is a simple black-box NOM baseline (works with NeuralFoil).
    """
    z = np.random.normal(loc=mu, scale=k * sigma)
    z = np.clip(z, mu - k * sigma, mu + k * sigma)
    return z.astype(float)


def evaluate_candidate(pipeline: TalarAIPipeline, latent_vec, alpha, Re):
    """
    Runs the pipeline and returns:
      CL, CD, coords, fix_mode
    """
    out = pipeline.eval_latent_with_neuralfoil(latent_vec, alpha=alpha, Re=Re)

    CL = float(out["CL"])
    CD = float(out["CD"])
    coords = out["coords"]
    fix_mode = out.get("fix_mode", "unknown")

    return CL, CD, coords, fix_mode


def nom_optimize(
    alpha: float = 6.0,
    Re: float = 5e5,
    n_iters: int = 500,
    k: float = 2.0,
    lam_bounds: float = 10.0,
    lam_geom: float = 50.0,
    min_thickness: float = 0.005,
    out_dir: str = "outputs",
):
    """
    NOM optimizer (black-box version):
    - propose latent vectors
    - evaluate via pipeline (NeuralFoil)
    - compute objective + ReLU penalties
    - keep best

    This matches your professor’s NOM notes:
      minimize [ CD/CL ] + Σ λ_i * ReLU(constraint violations)
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load dataset distribution for bounds + proposals
    latents = load_latent_dataset()
    mu = latents.mean(axis=0)
    sigma = latents.std(axis=0) + 1e-9

    lat_lo, lat_hi = make_default_latent_bounds(latents, k=k)

    pipeline = TalarAIPipeline()

    best = None  # dict storing best candidate
    history = []

    for it in range(1, n_iters + 1):
        candidate = propose_latent(mu, sigma, k=k)

        try:
            CL, CD, coords, fix_mode = evaluate_candidate(pipeline, candidate, alpha, Re)

            # Objective
            obj = default_objective(CL, CD)

            # Constraints (ReLU penalties)
            pen = total_penalty(
                latent_vec=candidate,
                coords=coords,
                lat_lo=lat_lo,
                lat_hi=lat_hi,
                lam_bounds=lam_bounds,
                lam_geom=lam_geom,
                min_thickness=min_thickness,
            )

            total = float(obj + pen)

            record = {
                "iter": it,
                "CL": float(CL),
                "CD": float(CD),
                "objective": float(obj),
                "penalty": float(pen),
                "total": float(total),
                "fix_mode": fix_mode,
            }
            history.append(record)

            if (best is None) or (total < best["total"]):
                best = {
                    **record,
                    "latent": candidate.copy(),
                    "coords": np.asarray(coords).copy(),
                }
                print(
                    f"[{it}/{n_iters}] New BEST total={best['total']:.6f} "
                    f"(obj={best['objective']:.6f}, pen={best['penalty']:.6f}) "
                    f"| CL={best['CL']:.4f} CD={best['CD']:.6f} fix={best['fix_mode']}"
                )

        except Exception as e:
            # Skip bad evaluations
            continue

    if best is None:
        raise RuntimeError("NOM failed: no valid candidates were evaluated.")

    # Save best artifacts
    np.save(out_path / "best_latent_nom.npy", best["latent"])
    np.savetxt(out_path / "best_coords_nom.csv", best["coords"], delimiter=",", header="x,y", comments="")

    # Save history + summary
    with open(out_path / "nom_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "alpha": alpha,
        "Re": Re,
        "n_iters": n_iters,
        "k": k,
        "lam_bounds": lam_bounds,
        "lam_geom": lam_geom,
        "min_thickness": min_thickness,
        "best_total": best["total"],
        "best_objective": best["objective"],
        "best_penalty": best["penalty"],
        "best_CL": best["CL"],
        "best_CD": best["CD"],
        "best_fix_mode": best["fix_mode"],
    }
    with open(out_path / "nom_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== NOM Finished ===")
    print(json.dumps(summary, indent=2))
    print("\nSaved outputs:")
    print("  outputs/best_latent_nom.npy")
    print("  outputs/best_coords_nom.csv")
    print("  outputs/nom_history.json")
    print("  outputs/nom_summary.json")


if __name__ == "__main__":
    # Default run (meeting-friendly)
    nom_optimize(
        alpha=6.0,
        Re=5e5,
        n_iters=500,
        k=2.0,
        lam_bounds=1.0,
        lam_geom=5.0,
        min_thickness=0.005,
        out_dir="outputs",
    )
