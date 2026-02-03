from __future__ import annotations

import numpy as np


def relu(x: float) -> float:
    return float(max(0.0, x))


def make_default_latent_bounds(latents: np.ndarray, k: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Bounds used for latent 'stay near training distribution' constraint.
    We use mean ± k*std (simple + explainable).
    """
    latents = np.asarray(latents, dtype=float)
    mu = np.nanmean(latents, axis=0)
    sig = np.nanstd(latents, axis=0) + 1e-9
    lo = mu - k * sig
    hi = mu + k * sig
    return lo, hi


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    """
    Normalize decoded coords so chord is ~[0,1] in x.
    This reduces numeric issues downstream.
    """
    c = np.asarray(coords, dtype=float)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError(f"coords must be (N,2). got {c.shape}")

    if not np.all(np.isfinite(c)):
        raise ValueError("coords contain NaN/Inf")

    x = c[:, 0]
    y = c[:, 1]

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    chord = x_max - x_min
    if chord <= 1e-12:
        raise ValueError("coords chord length too small/collapsed")

    x_n = (x - x_min) / chord
    c_n = np.column_stack([x_n, y])
    return c_n


def _min_thickness_estimate(coords_norm: np.ndarray, n_bins: int = 40) -> float:
    """
    Rough thickness estimate:
    - bin by x
    - thickness in each bin = (max y - min y)
    - min thickness = min over bins (excluding empty bins)

    Not perfect aero thickness, but good enough as a "geometry sanity" constraint.
    """
    x = coords_norm[:, 0]
    y = coords_norm[:, 1]

    # Ignore points outside chord due to bad decode
    mask = (x >= 0.0) & (x <= 1.0) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 10:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    t_vals = []
    for i in range(n_bins):
        m = (x >= bins[i]) & (x < bins[i + 1])
        if np.any(m):
            y_max = float(np.max(y[m]))
            y_min = float(np.min(y[m]))
            t_vals.append(y_max - y_min)

    if not t_vals:
        return 0.0

    t_vals = np.asarray(t_vals, dtype=float)
    t_vals = t_vals[np.isfinite(t_vals)]
    if t_vals.size == 0:
        return 0.0

    return float(np.min(t_vals))


def latent_bounds_penalty(latent_vec: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """
    Penalty for leaving latent box constraints.
    """
    z = np.asarray(latent_vec, dtype=float).reshape(-1)
    lo = np.asarray(lo, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)
    if z.shape != lo.shape or z.shape != hi.shape:
        raise ValueError("latent_vec and bounds must match shape")

    below = np.maximum(lo - z, 0.0)
    above = np.maximum(z - hi, 0.0)
    return float(np.sum(below + above))


def geometry_penalty(coords: np.ndarray, min_thickness: float = 0.005) -> tuple[float, dict]:
    """
    Geometry sanity penalties:
    - coords finite
    - chord not collapsed
    - min thickness >= threshold
    """
    c_n = _normalize_coords(coords)
    t_min = _min_thickness_estimate(c_n, n_bins=40)

    pen_t = relu(min_thickness - t_min)

    info = {
        "min_thickness_est": float(t_min),
    }
    return float(pen_t), info


def total_penalty(
    latent_vec: np.ndarray,
    coords: np.ndarray,
    lat_lo: np.ndarray,
    lat_hi: np.ndarray,
    lam_bounds: float = 1.0,
    lam_geom: float = 5.0,
    min_thickness: float = 0.005,
) -> float:
    """
    Total penalty = λ_bounds * P_bounds + λ_geom * P_geom

    P_bounds: latent box constraint
    P_geom: thickness/geometry sanity
    """
    p_bounds = latent_bounds_penalty(latent_vec, lat_lo, lat_hi)
    p_geom, _ = geometry_penalty(coords, min_thickness=min_thickness)

    return float(lam_bounds * p_bounds + lam_geom * p_geom)
