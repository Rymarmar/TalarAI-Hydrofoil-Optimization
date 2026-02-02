import numpy as np


def relu(x: float) -> float:
    """ReLU penalty: max(x, 0)."""
    return float(np.maximum(x, 0.0))


def bound_penalty(x: float, lo: float, hi: float) -> float:
    """
    Penalty for lo <= x <= hi:
      ReLU(x - hi) + ReLU(lo - x)
    """
    return relu(x - hi) + relu(lo - x)


def vector_bounds_penalty(vec, lo_vec, hi_vec) -> float:
    """
    Sum of bound penalties across a vector.
    vec, lo_vec, hi_vec should be same length.
    """
    v = np.asarray(vec, dtype=float)
    lo = np.asarray(lo_vec, dtype=float)
    hi = np.asarray(hi_vec, dtype=float)
    if v.shape != lo.shape or v.shape != hi.shape:
        raise ValueError("vector_bounds_penalty: shape mismatch.")
    total = 0.0
    for i in range(len(v)):
        total += bound_penalty(v[i], lo[i], hi[i])
    return float(total)


def geometry_penalty(coords, min_thickness: float = 0.005) -> float:
    """
    Optional geometry penalties.
    coords is (80,2): first 40 upper (x 0->1), last 40 lower reversed (x 1->0).

    We penalize:
      - upper below lower at matched x (basic validity)
      - too-thin airfoil (min thickness constraint)
    """
    c = np.asarray(coords, dtype=float)
    if c.shape[0] < 80 or c.shape[1] != 2:
        # If you ever change points count, update this.
        return 0.0

    n = 40
    upper = c[:n, :]
    lower = c[n:, :][::-1, :]  # reverse lower back to x 0->1 alignment

    y_upper = upper[:, 1]
    y_lower = lower[:, 1]

    thickness = y_upper - y_lower  # should be >= 0 ideally

    # Penalize crossings (upper below lower)
    crossing_pen = float(np.sum(np.maximum(-thickness, 0.0)))

    # Penalize min thickness violation
    tmin = float(np.min(thickness))
    thin_pen = relu(min_thickness - tmin)

    return crossing_pen + thin_pen


def make_default_latent_bounds(latents: np.ndarray, k: float = 2.0):
    """
    Compute dataset-based bounds: mean ± k*std for each parameter.
    This keeps NOM inside the learned latent distribution.
    """
    mu = np.mean(latents, axis=0)
    sigma = np.std(latents, axis=0) + 1e-9
    lo = mu - k * sigma
    hi = mu + k * sigma
    return lo, hi


def total_penalty(
    latent_vec,
    coords,
    lat_lo,
    lat_hi,
    lam_bounds: float = 1.0,
    lam_geom: float = 1.0,
    min_thickness: float = 0.005,
) -> float:
    """
    Total penalty = λ_bounds * (latent bounds violations) + λ_geom * (geometry violations)

    This matches your professor’s board idea:
      objective + λ1*ReLU(...) + λ2*ReLU(...) + ...
    """
    p_bounds = vector_bounds_penalty(latent_vec, lat_lo, lat_hi)
    p_geom = geometry_penalty(coords, min_thickness=min_thickness)

    return float(lam_bounds * p_bounds + lam_geom * p_geom)
