# Run:
#   python -m optimization.nom_driver
#
# Purpose:
#   Implement NOM (Neural Optimization Machine) loop:
#     - propose a latent design (6 params)
#     - decode + evaluate via NeuralFoil
#     - compute objective (performance)
#     - compute penalties (constraints)
#     - keep best design found
#     - save best latent + best coordinates + history
#
#   This is BLACK-BOX optimization (NeuralFoil isn't differentiable here),
#   so "learning rate" is a step size used for local proposals in latent space

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.talarai_pipeline import TalarAIPipeline
from optimization.objective import default_objective
from optimization.constraints import make_default_latent_bounds, total_penalty


def load_latent_dataset(csv_path: str = "data/airfoil_latent_params.csv") -> np.ndarray:
    """
    Load the existing latent dataset (training distribution reference).
    We use this ONLY to compute mean/std and bounds — not to "train" in NOM.
    """
    df = pd.read_csv(csv_path)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] != 6:
        raise ValueError(f"Expected 6 numeric latent columns, found {numeric_df.shape[1]} in {csv_path}")
    return numeric_df.values.astype(float)


def _as_scalar(x):
    """Convert numpy/array outputs to a float."""
    if isinstance(x, (list, tuple, np.ndarray)):
        x = np.asarray(x).reshape(-1)
        if x.size == 0:
            return float("nan")
        return float(x[0])
    return float(x)


def safe_eval(pipeline: TalarAIPipeline, latent_vec: np.ndarray, alpha: float, Re: float):
    """
    Evaluate a candidate latent design safely.
    Returns None if invalid (NaNs, weird values, broken coords).
    """
    try:
        out = pipeline.eval_latent_with_neuralfoil(latent_vec, alpha=alpha, Re=Re)
        CL = _as_scalar(out.get("CL"))
        CD = _as_scalar(out.get("CD"))
        coords = np.asarray(out.get("coords"), dtype=float)
        fix_mode = out.get("fix_mode", "unknown")

        # basic validity checks
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


def propose_latent_global(mu: np.ndarray, sigma: np.ndarray, k: float, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """
    GLOBAL proposal:
      sample around the dataset mean with spread k*sigma
      then clip to bounds
    This explores the overall design space.
    """
    z = np.random.normal(loc=mu, scale=k * sigma).astype(float)
    return np.clip(z, lo, hi).astype(float)


def propose_latent_local(center: np.ndarray, lr: float, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """
    LOCAL proposal:
      z = center + lr * random_step
    This exploits/refines around the current best.
    Here lr is the "learning rate" / step size in latent space.
    """
    step = np.random.normal(loc=0.0, scale=1.0, size=center.shape).astype(float)
    z = np.asarray(center, dtype=float) + float(lr) * step
    return np.clip(z, lo, hi).astype(float)


def _save_latent_csv(path: Path, latent: np.ndarray, pretty_sigfigs: int | None = None):
    """
    Save the 6 optimized latent parameters.
    We keep full precision internally; "pretty" is for reporting.
    """
    z = np.asarray(latent, dtype=float).reshape(-1)
    if z.size != 6:
        raise ValueError(f"Expected 6 latent params, got {z.size}")

    if pretty_sigfigs is None:
        vals = z
    else:
        vals = np.array([float(f"{v:.{pretty_sigfigs}g}") for v in z], dtype=float)

    df = pd.DataFrame([vals], columns=[f"p{i+1}" for i in range(6)])
    df.to_csv(path, index=False)


def _save_coords_csv(path: Path, coords: np.ndarray, pretty_decimals: int | None = None):
    """
    Save decoded airfoil coordinates.
    Again: full precision for real use, rounded for reporting.
    """
    c = np.asarray(coords, dtype=float)
    if pretty_decimals is not None:
        c = np.round(c, int(pretty_decimals))
    np.savetxt(path, c, delimiter=",", header="x,y", comments="")


def nom_optimize(
    alpha: float = 6.0,
    Re: float = 5e5,
    n_iters: int = 2000,
    k: float = 2.0,
    learning_rate_init: float = 0.25,
    lr_decay: float = 0.999,
    p_local: float = 0.75,
    lam_bounds: float = 1.0,
    lam_geom: float = 5.0,
    lam_cl: float = 10.0,
    min_thickness: float = 0.005,
    cl_min: float | None = 0.2,
    cl_max: float | None = 1.2,
    out_dir: str = "outputs",
):
    """
    Main NOM loop:
      minimize total = objective(CL,CD) + penalty(constraints)
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1) Load dataset to define bounds / distribution reference
    latents = load_latent_dataset()
    mu = latents.mean(axis=0)
    sigma = latents.std(axis=0) + 1e-9
    lat_lo, lat_hi = make_default_latent_bounds(latents, k=k)

    # 2) Create evaluation pipeline
    pipeline = TalarAIPipeline()

    best = None
    history = []
    valid = 0
    skipped = 0

    # "learning rate" = local step size in latent space
    lr = float(learning_rate_init)

    for it in range(1, n_iters + 1):
        # 3) Propose a candidate latent design
        use_local = (best is not None) and (np.random.rand() < float(p_local))
        if use_local:
            cand = propose_latent_local(best["latent"], lr=lr, lo=lat_lo, hi=lat_hi)
            mode = "local"
        else:
            cand = propose_latent_global(mu, sigma, k=k, lo=lat_lo, hi=lat_hi)
            mode = "global"

        # 4) Evaluate candidate with NeuralFoil
        res = safe_eval(pipeline, cand, alpha=alpha, Re=Re)
        if res is None:
            skipped += 1
            lr *= float(lr_decay)
            continue

        valid += 1
        CL = res["CL"]
        CD = res["CD"]
        coords = res["coords"]

        # 5) Compute performance objective (no constraints here)
        obj = default_objective(CL, CD)

        # 6) Compute constraint penalties (NOM-style ReLU penalties)
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

        # record this iteration (for plotting later)
        rec = {
            "iter": int(it),
            "CL": float(CL),
            "CD": float(CD),
            "objective": float(obj),
            "penalty": float(pen),
            "total": float(total),
            "tmin": float(pen_info.get("min_thickness_est", 0.0)),
            "lr": float(lr),
            "mode": str(mode),
        }
        history.append(rec)

        # 7) Keep best solution seen so far (simple greedy best-so-far)
        if best is None or total < best["total"]:
            best = {**rec, "latent": cand.copy(), "coords": coords.copy()}
            print(
                f"[{it}/{n_iters}] BEST total={best['total']:.6f} (obj={best['objective']:.6f}, pen={best['penalty']:.6f}) "
                f"| CL={best['CL']:.4f} CD={best['CD']:.6f} | tmin={best['tmin']:.5f} | lr={best['lr']:.4f} mode={best['mode']}"
            )

        # 8) Decay step size to refine search over time
        lr *= float(lr_decay)

    if best is None:
        raise RuntimeError("NOM failed: 0 valid candidates.")

    # 9) Save best artifacts for downstream use (CAD/rebuild/plots)
    np.save(out_path / "best_latent_nom.npy", best["latent"])
    _save_latent_csv(out_path / "best_latent_nom.csv", best["latent"], pretty_sigfigs=None)
    _save_coords_csv(out_path / "best_coords_nom.csv", best["coords"], pretty_decimals=None)

    # Pretty versions for reporting/printing
    _save_latent_csv(out_path / "best_latent_nom_pretty.csv", best["latent"], pretty_sigfigs=3)
    _save_coords_csv(out_path / "best_coords_nom_pretty.csv", best["coords"], pretty_decimals=6)

    # Save full history
    with open(out_path / "nom_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Save summary (easy to cite in presentations)
    summary = {
        "alpha": float(alpha),
        "Re": float(Re),
        "n_iters": int(n_iters),
        "k": float(k),
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
        "best_latent_params": [float(x) for x in np.asarray(best["latent"], dtype=float).reshape(-1)],
    }

    with open(out_path / "nom_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== NOM Finished ===")
    print(json.dumps(summary, indent=2))
    print("\nSaved outputs in outputs/ (latent params, coords, history, summary).")


if __name__ == "__main__":
    nom_optimize()
