from __future__ import annotations
import numpy as np

# Purpose:
#   Convert "design requirements" into penalty terms that NOM can minimize
#
# From NOM reference:
#   total_cost = objective + λ1*ReLU(constraint1_violation) + λ2*ReLU(constraint2_violation) + ...
#
# Here we implement three types:
#   1) latent bounds (stay near training distribution)
#   2) geometry thickness (manufacturability sanity)
#   3) CL bounds (keep lift in a physically realistic / target range)

# ReLu means only violations get penalized; if constraints are satisfied, penalty = 0
def relu(x: float) -> float:
    """ReLU = max(0, x). Only penalize when constraint is violated."""
    return float(max(0.0, x))

# Some latent vectors can decode into unrealistic shapes and unstable aero predictions
def make_default_latent_bounds(latents: np.ndarray, k: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Build latent bounds from dataset: mean ± k*std
    keeps the optimizer from wandering too far from training data
    """
    latents = np.asarray(latents, dtype=float)
    mu = np.nanmean(latents, axis=0)
    sig = np.nanstd(latents, axis=0) + 1e-9
    lo = mu - k * sig
    hi = mu + k * sig
    return lo, hi

# fast proxy constraints used during optimization
def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    """
    Normalize coords so chord is ~[0,1] in x (proxy constraint)
    This makes thickness checks consistent across shapes
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
    return np.column_stack([x_n, y])


def _min_thickness_estimate(coords_norm: np.ndarray, n_bins: int = 40) -> float:
    """
    Rough thickness estimate:
      bin x into segments
      thickness in bin = max(y) - min(y)
      return the minimum thickness across bins
    This catches ultra-thin / degenerate shapes
    """
    x = coords_norm[:, 0]
    y = coords_norm[:, 1]

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
            t_vals.append(float(np.max(y[m])) - float(np.min(y[m])))

    if not t_vals:
        return 0.0

    t_vals = np.asarray(t_vals, dtype=float)
    t_vals = t_vals[np.isfinite(t_vals)]
    if t_vals.size == 0:
        return 0.0

    return float(np.min(t_vals))

# adds penaly if any latent component violates the box bounds
def latent_bounds_penalty(latent_vec: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """
    Penalize leaving the latent box constraints:
      sum( ReLU(lo - z) + ReLU(z - hi) )
    """
    z = np.asarray(latent_vec, dtype=float).reshape(-1)
    lo = np.asarray(lo, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)
    if z.shape != lo.shape or z.shape != hi.shape:
        raise ValueError("latent_vec and bounds must match shape")

    below = np.maximum(lo - z, 0.0)
    above = np.maximum(z - hi, 0.0)
    return float(np.sum(below + above))

# Turns thickness into a ReLU penalty
def geometry_penalty(coords: np.ndarray, min_thickness: float = 0.005) -> tuple[float, dict]:
    """
    Geometry constraint:
      penalize if estimated min thickness < min_thickness
    Returns:
      (penalty_value, info_dict)
    """
    c_n = _normalize_coords(coords)
    t_min = _min_thickness_estimate(c_n, n_bins=40)

    # ReLU violation: only positive when thickness is too small
    pen_t = relu(min_thickness - t_min)

    return float(pen_t), {"min_thickness_est": float(t_min)}

# Bounds CL to keep designs in a realistic operating envelope for physical testing
def cl_bounds_penalty(CL: float, cl_min: float | None = None, cl_max: float | None = None) -> float:
    """
    Lift coefficient constraint:
      if cl_min: ReLU(cl_min - CL)
      if cl_max: ReLU(CL - cl_max)
    """
    pen = 0.0
    if cl_min is not None:
        pen += relu(float(cl_min) - float(CL))
    if cl_max is not None:
        pen += relu(float(CL) - float(cl_max))
    return float(pen)

# Adds lambda weights to all constraints
def total_penalty(
    latent_vec: np.ndarray,
    coords: np.ndarray,
    CL: float | None,
    lat_lo: np.ndarray,
    lat_hi: np.ndarray,
    lam_bounds: float = 1.0,
    lam_geom: float = 5.0,
    lam_cl: float = 10.0,
    min_thickness: float = 0.005,
    cl_min: float | None = None,
    cl_max: float | None = None,
) -> tuple[float, dict]:
    """
    Combine all constraints into one penalty:
      P = λ_bounds*P_bounds + λ_geom*P_geom + λ_cl*P_cl
    """
    # 1) stay near training distribution
    p_bounds = latent_bounds_penalty(latent_vec, lat_lo, lat_hi)

    # 2) manufacturability sanity
    p_geom, geom_info = geometry_penalty(coords, min_thickness=min_thickness)

    # 3) keep lift in realistic / desired range
    p_cl = 0.0
    if CL is not None:
        p_cl = cl_bounds_penalty(CL, cl_min=cl_min, cl_max=cl_max)

    total = float(lam_bounds * p_bounds + lam_geom * p_geom + lam_cl * p_cl)

    info = {
        **geom_info,
        "p_bounds": float(p_bounds),
        "p_geom": float(p_geom),
        "p_cl": float(p_cl),
    }
    return total, info
