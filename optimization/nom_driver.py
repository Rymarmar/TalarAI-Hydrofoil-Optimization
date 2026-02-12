"""
python -m optimization.nom_driver

optimization/nom_driver.py

NOM loop (black-box optimization):
  - propose latent z (6 params)
  - pipeline -> decode -> NeuralFoil -> CL, CD
  - objective = CD/CL
  - penalty = lambdas * ReLU(violations)
  - minimize total = objective + penalty
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Import paths are a frequent source of "it works on my machine" issues.
# We support both:
#   1) running from repo root:    python optimization/nom_driver.py
#   2) running as a module:       python -m optimization.nom_driver
try:
    from pipeline.talarai_pipeline import TalarAIPipeline
except ModuleNotFoundError:  # pragma: no cover
    from talarai_pipeline import TalarAIPipeline

try:
    from optimization.objective import default_objective
    from optimization.constraints import latent_minmax_bounds, total_penalty
except ModuleNotFoundError:  # pragma: no cover
    from objective import default_objective
    from constraints import latent_minmax_bounds, total_penalty


def load_latent_dataset(csv_path: str = "data/airfoil_latent_params.csv") -> np.ndarray:
    df = pd.read_csv(csv_path)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] != 6:
        raise ValueError(f"Expected 6 numeric latent columns, got {numeric.shape[1]}")
    return numeric.values.astype(float)


def safe_eval(pipeline: TalarAIPipeline, latent_vec: np.ndarray, alpha: float, Re: float):
    """
    Reject invalid candidates early (cheap sanity checks):
      - NaNs
      - broken coords
      - insane CD
    """
    try:
        out = pipeline.eval_latent_with_neuralfoil(latent_vec, alpha=alpha, Re=Re)
        CL = float(out["CL"])
        CD = float(out["CD"])
        coords = np.asarray(out["coords"], dtype=float)

        if not np.isfinite(CL) or not np.isfinite(CD):
            return None
        if CD <= 0.0 or CD > 10.0:
            return None
        if coords.ndim != 2 or coords.shape[1] != 2:
            return None
        if not np.all(np.isfinite(coords)):
            return None

        return {"CL": CL, "CD": CD, "coords": coords}
    except Exception:
        return None


def propose_global(lat_lo: np.ndarray, lat_hi: np.ndarray) -> np.ndarray:
    """
    Global exploration:
      sample uniformly inside dataset min/max per param
    (Action item: no need for k / mean/std machinery)
    """
    return np.random.uniform(lat_lo, lat_hi).astype(float)


def propose_local(best_latent: np.ndarray, lr: float, lat_lo: np.ndarray, lat_hi: np.ndarray) -> np.ndarray:
    """
    Local refinement around best:
      z = best + lr * N(0,1)
    lr is the "step size" (Action item: 1e-3 or less)
    """
    step = np.random.normal(loc=0.0, scale=1.0, size=best_latent.shape).astype(float)
    z = np.asarray(best_latent, dtype=float) + float(lr) * step
    return np.clip(z, lat_lo, lat_hi).astype(float)


def nom_optimize(
    alpha: float = 6.0,
    Re: float = 5e5,
    n_iters: int = 2000,
    learning_rate_init: float = 1e-3,   # <= 1e-3 per action item
    lr_decay: float = 0.999,
    p_local: float = 0.75,              # probability we exploit locally vs explore globally
    # Lambdas (constraint weights)
    lam_bounds: float = 1.0,
    lam_geom: float = 5.0,
    lam_cl: float = 10.0,
    # Constraint thresholds
    min_thickness: float = 0.005,
    # Meeting note (CL bounds action item): set these to match YOUR operating point.
    # For the SkiCat numbers discussed:
    #   - CL_cruise ~ 0.268
    #   - CL_min_flying ~ 0.697
    #   - CL_vmax ~ 0.11
    # A reasonable first window for early NOM runs is [0.10, 0.80].
    # BEFORE CL REQUIREMENTS CALCULATIONS
    cl_min: float | None = 0.10,
    cl_max: float | None = 0.80,
    out_dir: str = "outputs",
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1) Latent bounds from dataset min/max per dimension
    latents = load_latent_dataset()
    lat_lo, lat_hi = latent_minmax_bounds(latents)

    # 2) Pipeline eval object
    pipeline = TalarAIPipeline()

    best = None
    history: list[dict] = []
    valid = 0
    skipped = 0

    lr = float(learning_rate_init)

    for it in range(1, n_iters + 1):
        # 3) Propose candidate (global vs local)
        use_local = (best is not None) and (np.random.rand() < float(p_local))
        if use_local:
            cand = propose_local(best["latent"], lr=lr, lat_lo=lat_lo, lat_hi=lat_hi)
            mode = "local"
        else:
            cand = propose_global(lat_lo, lat_hi)
            mode = "global"

        # 4) Evaluate candidate
        res = safe_eval(pipeline, cand, alpha=alpha, Re=Re)
        if res is None:
            skipped += 1
            lr *= float(lr_decay)
            continue

        valid += 1
        CL = res["CL"]
        CD = res["CD"]
        coords = res["coords"]

        # 5) Objective (performance only)
        obj = default_objective(CL, CD)

        # 6) Penalties (constraints)
        pen, pen_info = total_penalty(
            latent_vec=cand,
            coords=coords,
            CL=CL,
            lat_lo=lat_lo,
            lat_hi=lat_hi,
            lam_bounds=lam_bounds,
            lam_geom=lam_geom,
            lam_cl=lam_cl,
            min_thickness=min_thickness,
            cl_min=cl_min,
            cl_max=cl_max,
        )

        total = float(obj + pen)

        rec = {
            "iter": int(it),
            "mode": mode,
            "lr": float(lr),
            "CL": float(CL),
            "CD": float(CD),
            "objective": float(obj),
            "penalty": float(pen),
            "total": float(total),
            "tmin": float(pen_info.get("min_thickness_est", 0.0)),
        }
        history.append(rec)

        # 7) Keep best (greedy best-so-far)
        if best is None or total < best["total"]:
            best = {**rec, "latent": cand.copy(), "coords": coords.copy()}
            print(
                f"[{it}/{n_iters}] BEST total={best['total']:.6f} "
                f"(obj={best['objective']:.6f}, pen={best['penalty']:.6f}) "
                f"| CL={best['CL']:.4f} CD={best['CD']:.6f} "
                f"| tmin={best['tmin']:.6f} lr={best['lr']:.3e} mode={best['mode']}"
            )

        # 8) Decay local step size
        lr *= float(lr_decay)

    if best is None:
        raise RuntimeError("NOM failed: 0 valid candidates.")

    # 9) Save artifacts
    np.save(out_path / "best_latent_nom.npy", best["latent"])
    np.savetxt(out_path / "best_coords_nom.csv", best["coords"], delimiter=",", header="x,y", comments="")

    with open(out_path / "nom_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "alpha": float(alpha),
        "Re": float(Re),
        "n_iters": int(n_iters),
        "learning_rate_init": float(learning_rate_init),
        "lr_decay": float(lr_decay),
        "p_local": float(p_local),
        "lam_bounds": float(lam_bounds),
        "lam_geom": float(lam_geom),
        "lam_cl": float(lam_cl),
        "min_thickness": float(min_thickness),
        "cl_min": None if cl_min is None else float(cl_min),
        "cl_max": None if cl_max is None else float(cl_max),
        "valid_evals": int(valid),
        "skipped": int(skipped),
        "best_total": float(best["total"]),
        "best_objective": float(best["objective"]),
        "best_penalty": float(best["penalty"]),
        "best_CL": float(best["CL"]),
        "best_CD": float(best["CD"]),
        "best_min_thickness_est": float(best["tmin"]),
        "best_latent_params": [float(x) for x in np.asarray(best["latent"]).reshape(-1)],
        "latent_lo": [float(x) for x in np.asarray(lat_lo).reshape(-1)],
        "latent_hi": [float(x) for x in np.asarray(lat_hi).reshape(-1)],
    }

    with open(out_path / "nom_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== NOM Finished ===")
    print(json.dumps(summary, indent=2))
    print("\nSaved outputs in outputs/.")


if __name__ == "__main__":
    nom_optimize()