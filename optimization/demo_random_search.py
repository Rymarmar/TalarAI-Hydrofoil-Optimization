import numpy as np
import pandas as pd

from pipeline.talarai_pipeline import TalarAIPipeline


def _is_finite(x: float) -> bool:
    return np.isfinite(x) and not np.isnan(x)


def _score_cd_over_abscl(CL: float, CD: float, eps: float = 1e-9) -> float:
    """Objective: minimize CD/|CL| (robust if CL sign flips)."""
    return CD / max(abs(CL), eps)


def _basic_sanity(CL: float, CD: float, cd_max: float = 5.0) -> bool:
    """
    Loose sanity filter to avoid complete nonsense while still allowing the script to run.
    - finite values
    - CD positive
    - CD not astronomically huge (default 5.0 is very loose)
    - |CL| not tiny
    """
    if not (_is_finite(CL) and _is_finite(CD)):
        return False
    if CD <= 0 or CD > cd_max:
        return False
    if abs(CL) < 1e-6:
        return False
    return True


def _load_latents() -> np.ndarray:
    df = pd.read_csv("data/airfoil_latent_params.csv")
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] != 6:
        raise ValueError(
            f"Expected 6 numeric latent columns, found {numeric_df.shape[1]}. "
            "Check data/airfoil_latent_params.csv format."
        )

    return numeric_df.values.astype(float)


def _select_baseline_best_available(
    pipeline: TalarAIPipeline,
    latents: np.ndarray,
    alpha: float,
    Re: float,
    max_checks: int = 500,
    cd_max_sanity: float = 5.0,
):
    """
    Always selects a baseline by scanning candidates and taking the BEST score (CD/|CL|)
    among valid finite outputs. Prefers CL>0 if possible, but will fall back to CL<0 if needed.

    Returns: (latent, out, idx, stats_dict)
    """
    checks = min(max_checks, len(latents))

    best_any = None  # (score, latent, out, idx)
    best_posCL = None

    valid = 0
    skipped = 0

    for i in range(checks):
        latent = latents[i]
        try:
            out = pipeline.eval_latent_with_neuralfoil(latent, alpha=alpha, Re=Re)
            CL = float(out["CL"])
            CD = float(out["CD"])

            if not _basic_sanity(CL, CD, cd_max=cd_max_sanity):
                skipped += 1
                continue

            valid += 1
            score = _score_cd_over_abscl(CL, CD)

            if (best_any is None) or (score < best_any[0]):
                best_any = (score, latent, out, i)

            if CL > 0 and ((best_posCL is None) or (score < best_posCL[0])):
                best_posCL = (score, latent, out, i)

        except Exception:
            skipped += 1
            continue

    stats = {"scanned": checks, "valid": valid, "skipped": skipped}

    if best_posCL is not None:
        score, latent, out, idx = best_posCL
        stats["baseline_choice"] = "best_positive_CL"
        return latent, out, idx, score, stats

    if best_any is not None:
        score, latent, out, idx = best_any
        stats["baseline_choice"] = "best_any_sign"
        return latent, out, idx, score, stats

    raise RuntimeError(
        f"No valid baselines found in first {checks} candidates. "
        "This usually means NeuralFoil is returning non-finite values for everything. "
        "Check geometry validity / coordinate convention."
    )


def main():
    print("=== Demo Random Search (Sponsor-safe): TalarAI ===")

    alpha_deg = 6.0
    Re = 5e5

    pipeline = TalarAIPipeline()

    latents = _load_latents()
    print(f"Loaded latent dataset: {latents.shape[0]} rows, {latents.shape[1]} params")

    # Distribution sampling setup
    mu = latents.mean(axis=0)
    sigma = latents.std(axis=0) + 1e-9
    k = 2.0

    # Baseline selection that ALWAYS works (best available)
    baseline_latent, baseline_out, baseline_idx, baseline_score, stats = _select_baseline_best_available(
        pipeline, latents, alpha=alpha_deg, Re=Re, max_checks=800, cd_max_sanity=5.0
    )

    bCL = float(baseline_out["CL"])
    bCD = float(baseline_out["CD"])

    print("\nBaseline (auto-selected):")
    print(f"  scanned = {stats['scanned']} | valid = {stats['valid']} | skipped = {stats['skipped']}")
    print(f"  baseline_choice = {stats['baseline_choice']}")
    print(f"  baseline_row = {baseline_idx}")
    print(f"  fix_mode = {baseline_out.get('fix_mode', 'unknown')}")
    print(f"  CL = {bCL:.6f}")
    print(f"  CD = {bCD:.8f}")
    print(f"  CD/|CL| = {baseline_score:.8f}")

    # Random search sampling (still sanity filtered)
    n_samples = 300
    cd_max_sanity = 5.0

    best_latent = baseline_latent
    best_out = baseline_out
    best_score = baseline_score

    valid_count = 0
    skipped_count = 0

    print(f"\nRunning distribution-based random search: n={n_samples}, k={k} std\n")

    for i in range(n_samples):
        z = np.random.normal(loc=mu, scale=k * sigma)
        z = np.clip(z, mu - k * sigma, mu + k * sigma)

        try:
            out = pipeline.eval_latent_with_neuralfoil(z, alpha=alpha_deg, Re=Re)
            CL = float(out["CL"])
            CD = float(out["CD"])

            if not _basic_sanity(CL, CD, cd_max=cd_max_sanity):
                skipped_count += 1
                continue

            valid_count += 1
            score = _score_cd_over_abscl(CL, CD)

            if score < best_score:
                best_score = score
                best_latent = z
                best_out = out
                print(f"[{i+1}/{n_samples}] New best! CD/|CL| = {best_score:.8f} | CL={CL:.4f} CD={CD:.6f}")

        except Exception:
            skipped_count += 1
            continue

    best_CL = float(best_out["CL"])
    best_CD = float(best_out["CD"])

    print("\n=== Summary ===")
    print(f"Valid evals: {valid_count} | Skipped: {skipped_count}")

    print("\n=== Best Result ===")
    print(f"fix_mode = {best_out.get('fix_mode', 'unknown')}")
    print(f"Best latent params: {best_latent}")
    print(f"CL = {best_CL:.6f}")
    print(f"CD = {best_CD:.8f}")
    print(f"CD/|CL| = {best_score:.8f}")

    # Save outputs
    np.save("outputs/baseline_latent.npy", baseline_latent)
    np.savetxt("outputs/baseline_coords.csv", baseline_out["coords"], delimiter=",", header="x,y", comments="")

    np.save("outputs/best_latent_random.npy", best_latent)
    np.savetxt("outputs/best_coords_random.csv", best_out["coords"], delimiter=",", header="x,y", comments="")

    print("\nSaved outputs:")
    print("  outputs/baseline_latent.npy")
    print("  outputs/baseline_coords.csv")
    print("  outputs/best_latent_random.npy")
    print("  outputs/best_coords_random.csv")


if __name__ == "__main__":
    main()
