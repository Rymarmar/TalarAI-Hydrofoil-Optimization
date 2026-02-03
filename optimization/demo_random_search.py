from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from pipeline.talarai_pipeline import TalarAIPipeline
from optimization.objective import default_objective


def load_latents(csv_path: str = "data/airfoil_latent_params.csv") -> np.ndarray:
    df = pd.read_csv(csv_path)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] != 6:
        raise ValueError(f"Expected 6 numeric columns in {csv_path}, got {numeric.shape[1]}")
    return numeric.values.astype(float)


def _as_scalar(x):
    # Handles cases where pipeline returns np arrays (shape (1,) etc.)
    if isinstance(x, (list, tuple, np.ndarray)):
        x = np.asarray(x).reshape(-1)
        if x.size == 0:
            return float("nan")
        return float(x[0])
    return float(x)


def safe_eval(pipeline: TalarAIPipeline, latent: np.ndarray, alpha: float, Re: float):
    """
    Evaluate a latent; return dict or None if invalid.
    """
    try:
        out = pipeline.eval_latent_with_neuralfoil(latent, alpha=alpha, Re=Re)
        CL = _as_scalar(out.get("CL"))
        CD = _as_scalar(out.get("CD"))
        coords = np.asarray(out.get("coords"), dtype=float)
        fix_mode = out.get("fix_mode", "unknown")

        # basic numeric checks
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


def choose_baseline(
    pipeline: TalarAIPipeline,
    latents: np.ndarray,
    alpha: float,
    Re: float,
    max_checks: int = 800,
):
    """
    Scan dataset rows and pick a baseline that:
    - evaluates cleanly
    - prefers positive CL (designer-friendly)
    - otherwise falls back to best objective regardless of sign
    """
    max_checks = min(max_checks, latents.shape[0])

    best_pos = None
    best_any = None

    valid = 0
    skipped = 0

    for i in range(max_checks):
        res = safe_eval(pipeline, latents[i], alpha, Re)
        if res is None:
            skipped += 1
            continue

        valid += 1
        obj = default_objective(res["CL"], res["CD"])

        cand = {
            "idx": i,
            "latent": latents[i].copy(),
            "CL": res["CL"],
            "CD": res["CD"],
            "obj": float(obj),
            "coords": res["coords"].copy(),
            "fix_mode": res["fix_mode"],
        }

        if best_any is None or cand["obj"] < best_any["obj"]:
            best_any = cand

        if cand["CL"] > 0:
            if best_pos is None or cand["obj"] < best_pos["obj"]:
                best_pos = cand

    # Prefer positive CL if we found one
    baseline = best_pos if best_pos is not None else best_any
    if baseline is None:
        raise RuntimeError("Could not find any valid baseline in dataset scan.")

    baseline["scan_stats"] = {"scanned": max_checks, "valid": valid, "skipped": skipped}
    baseline["baseline_choice"] = "best_positive_CL" if best_pos is not None else "best_any_sign"
    return baseline


def propose_around(baseline: np.ndarray, sigma: np.ndarray, k: float = 2.0) -> np.ndarray:
    """
    Distribution-based random search:
    z = baseline + Normal(0, k*sigma)
    """
    z = baseline + np.random.normal(loc=0.0, scale=k * sigma)
    return z.astype(float)


def save_outputs(out_dir: str, baseline, best):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    np.save(out_path / "baseline_latent.npy", baseline["latent"])
    np.savetxt(out_path / "baseline_coords.csv", baseline["coords"], delimiter=",", header="x,y", comments="")

    np.save(out_path / "best_latent_random.npy", best["latent"])
    np.savetxt(out_path / "best_coords_random.csv", best["coords"], delimiter=",", header="x,y", comments="")


def main():
    print("=== Demo Random Search (Sponsor-safe): TalarAI ===")

    alpha_deg = 6.0
    Re = 5e5

    latents = load_latents("data/airfoil_latent_params.csv")
    print(f"Loaded latent dataset: {latents.shape[0]} rows, {latents.shape[1]} params")

    mu = latents.mean(axis=0)
    sigma = latents.std(axis=0) + 1e-9

    pipeline = TalarAIPipeline()

    baseline = choose_baseline(pipeline, latents, alpha=alpha_deg, Re=Re, max_checks=800)

    ss = baseline["scan_stats"]
    print("\nBaseline (auto-selected):")
    print(f"  scanned = {ss['scanned']} | valid = {ss['valid']} | skipped = {ss['skipped']}")
    print(f"  baseline_choice = {baseline['baseline_choice']}")
    print(f"  baseline_row = {baseline['idx']}")
    print(f"  fix_mode = {baseline['fix_mode']}")
    print(f"  CL = {baseline['CL']:.6f}")
    print(f"  CD = {baseline['CD']:.8f}")
    print(f"  CD/|CL| = {baseline['obj']:.8f}")

    # Random search params
    n = 300
    k = 2.0
    print(f"\nRunning distribution-based random search: n={n}, k={k:.1f} std\n")

    best = baseline.copy()
    valid_evals = 0
    skipped = 0

    for i in range(1, n + 1):
        cand_lat = propose_around(baseline["latent"], sigma=sigma, k=k)
        res = safe_eval(pipeline, cand_lat, alpha=alpha_deg, Re=Re)
        if res is None:
            skipped += 1
            continue

        valid_evals += 1
        obj = default_objective(res["CL"], res["CD"])

        if obj < best["obj"]:
            best = {
                "idx": None,
                "latent": cand_lat.copy(),
                "CL": res["CL"],
                "CD": res["CD"],
                "obj": float(obj),
                "coords": res["coords"].copy(),
                "fix_mode": res["fix_mode"],
            }
            print(
                f"[{i}/{n}] New best! CD/|CL|={best['obj']:.8f} | "
                f"CL={best['CL']:.4f} CD={best['CD']:.6f}"
            )

    print("\n=== Summary ===")
    print(f"Valid evals: {valid_evals} | skipped: {skipped}")

    print("\n=== Best Result ===")
    print(f"fix_mode = {best['fix_mode']}")
    print(f"Best latent params: {best['latent']}")
    print(f"CL = {best['CL']:.6f}")
    print(f"CD = {best['CD']:.8f}")
    print(f"CD/|CL| = {best['obj']:.8f}")

    save_outputs("outputs", baseline, best)

    print("\nSaved outputs:")
    print("  outputs/baseline_latent.npy")
    print("  outputs/baseline_coords.csv")
    print("  outputs/best_latent_random.npy")
    print("  outputs/best_coords_random.csv")


if __name__ == "__main__":
    main()
