# python -m optimization.nom_driver

from __future__ import annotations

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


def _as_scalar(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        x = np.asarray(x).reshape(-1)
        if x.size == 0:
            return float("nan")
        return float(x[0])
    return float(x)


def safe_eval(pipeline: TalarAIPipeline, latent_vec: np.ndarray, alpha: float, Re: float):
    """
    Evaluate latent -> returns dict or None if invalid.
    """
    try:
        out = pipeline.eval_latent_with_neuralfoil(latent_vec, alpha=alpha, Re=Re)
        CL = _as_scalar(out.get("CL"))
        CD = _as_scalar(out.get("CD"))
        coords = np.asarray(out.get("coords"), dtype=float)
        fix_mode = out.get("fix_mode", "unknown")

        if not np.isfinite(CL) or not np.isfinite(CD):
            return None
        if CD <= 0.0 or CD > 10.0:
            return None
        if coords.ndim != 2 or coords.shape[1] != 2:
            return None
        if not np.all(np.isfinite(coords)):
            return None

        return {"CL": CL, "CD": CD, "coords": coords, "fix_mode": fix_mode}
    except Exception:
        return None


def propose_latent(mu: np.ndarray, sigma: np.ndarray, k: float = 2.0, lo=None, hi=None) -> np.ndarray:
    """
    NOM proposal step (black-box):
    - sample Normal(mu, k*sigma)
    - optionally clip to latent bounds
    """
    z = np.random.normal(loc=mu, scale=k * sigma).astype(float)
    if lo is not None and hi is not None:
        z = np.clip(z, lo, hi)
    return z.astype(float)


def nom_optimize(
    alpha: float = 6.0,
    Re: float = 5e5,
    n_iters: int = 500,
    k: float = 2.0,
    lam_bounds: float = 1.0,
    lam_geom: float = 5.0,
    min_thickness: float = 0.005,
    out_dir: str = "outputs",
):
    """
    NOM (Neural Optimization Machine) — black-box version:

      minimize:
        objective(CL, CD) + penalty(latent bounds + geometry)

    objective defaults to CD/|CL| (sponsor-safe).

    penalty uses ReLU-style constraints:
      - latent stays near training distribution
      - geometry is not degenerate (thickness sanity)
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    latents = load_latent_dataset()
    mu = latents.mean(axis=0)
    sigma = latents.std(axis=0) + 1e-9

    lat_lo, lat_hi = make_default_latent_bounds(latents, k=k)

    pipeline = TalarAIPipeline()

    best = None
    history = []
    valid = 0
    skipped = 0

    for it in range(1, n_iters + 1):
        cand = propose_latent(mu, sigma, k=k, lo=lat_lo, hi=lat_hi)

        res = safe_eval(pipeline, cand, alpha=alpha, Re=Re)
        if res is None:
            skipped += 1
            continue

        valid += 1

        CL = res["CL"]
        CD = res["CD"]
        coords = res["coords"]
        fix_mode = res["fix_mode"]

        obj = default_objective(CL, CD)

        pen = total_penalty(
            latent_vec=cand,
            coords=coords,
            lat_lo=lat_lo,
            lat_hi=lat_hi,
            lam_bounds=lam_bounds,
            lam_geom=lam_geom,
            min_thickness=min_thickness,
        )

        total = float(obj + pen)

        rec = {
            "iter": it,
            "CL": float(CL),
            "CD": float(CD),
            "objective": float(obj),
            "penalty": float(pen),
            "total": float(total),
            "fix_mode": fix_mode,
        }
        history.append(rec)

        if best is None or total < best["total"]:
            best = {
                **rec,
                "latent": cand.copy(),
                "coords": coords.copy(),
            }
            print(
                f"[{it}/{n_iters}] New BEST total={best['total']:.6f} "
                f"(obj={best['objective']:.6f}, pen={best['penalty']:.6f}) "
                f"| CL={best['CL']:.4f} CD={best['CD']:.6f} fix={best['fix_mode']}"
            )

    if best is None:
        raise RuntimeError(
            f"NOM failed: 0 valid candidates. Try increasing n_iters, increasing k, "
            f"or relaxing geometry constraints."
        )

    # Save best artifacts
    np.save(out_path / "best_latent_nom.npy", best["latent"])
    np.savetxt(out_path / "best_coords_nom.csv", best["coords"], delimiter=",", header="x,y", comments="")

    with open(out_path / "nom_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "alpha": float(alpha),
        "Re": float(Re),
        "n_iters": int(n_iters),
        "k": float(k),
        "lam_bounds": float(lam_bounds),
        "lam_geom": float(lam_geom),
        "min_thickness": float(min_thickness),
        "valid_evals": int(valid),
        "skipped": int(skipped),
        "best_total": float(best["total"]),
        "best_objective": float(best["objective"]),
        "best_penalty": float(best["penalty"]),
        "best_CL": float(best["CL"]),
        "best_CD": float(best["CD"]),
        "best_fix_mode": str(best["fix_mode"]),
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
    # Meeting-friendly defaults
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

# #alpha → “Angle of attack used during evaluation”

# Re → “Reynolds number used during evaluation”

# n_iters → “Number of optimization iterations”

# k → “Controls how far NOM samples around the baseline distribution”

# lam_bounds → “Penalty weight for leaving the latent design space”

# lam_geom → “Penalty weight for violating geometric constraints”

# min_thickness → “Minimum allowable foil thickness”

# valid_evals → “Number of candidates that produced valid evaluations”

# skipped → “Candidates discarded due to invalid geometry or outputs”

# best_total → “Objective plus penalties for the best solution”

# best_objective → “Aerodynamic objective value only”

# best_penalty → “Constraint violation penalty (zero means fully valid)”

# best_CL / best_CD → “Predicted lift and drag for the optimized foil”

# best_fix_mode → “Internal geometry orientation correction”