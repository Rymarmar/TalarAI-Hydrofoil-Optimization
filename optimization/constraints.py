from __future__ import annotations
import numpy as np

"""
constraints.py

NOM penalty structure:
  total_cost = objective + λ_bounds*ReLU(bounds violation) + λ_geom*ReLU(geometry violation) + λ_cl*ReLU(CL violation)

We implement:
  1) latent bounds penalty (stay inside training min/max for each latent parameter)
  2) geometry penalty (min thickness must be above threshold; avoids degenerate foils)
  3) CL bounds penalty (keep lift in realistic/target range)

Important:
- We do NOT "normalize chords" here because our pipeline already outputs x in [0,1] (and uses .dat order).
"""


def relu(x: float) -> float:
    return float(max(0.0, x))


# ---------- Latent bounds (min/max per dimension) ----------

def latent_minmax_bounds(latents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Action item: "Find min and max of each param" (no mean±kstd).
    latents: (N,6)
    """
    Z = np.asarray(latents, dtype=float)
    if Z.ndim != 2 or Z.shape[1] != 6:
        raise ValueError(f"Expected latents shape (N,6), got {Z.shape}")

    lo = np.nanmin(Z, axis=0)
    hi = np.nanmax(Z, axis=0)

    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("Latent bounds contain NaN/Inf (bad dataset?)")

    return lo.astype(float), hi.astype(float)


def latent_bounds_penalty(latent_vec: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """
    NOM-style box constraint:
      sum( ReLU(lo - z) + ReLU(z - hi) )
    """
    z = np.asarray(latent_vec, dtype=float).reshape(-1)
    lo = np.asarray(lo, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)

    if z.shape != lo.shape or z.shape != hi.shape:
        raise ValueError(f"latent_vec and bounds must match: z={z.shape} lo={lo.shape} hi={hi.shape}")

    below = np.maximum(lo - z, 0.0)
    above = np.maximum(z - hi, 0.0)
    return float(np.sum(below + above))


# ---------- Geometry: min thickness (TE-safe) ----------

def _min_thickness_estimate(coords: np.ndarray, n_bins: int = 40, x_min: float = 0.05, x_max: float = 0.95) -> float:
    """
    Thickness proxy:
      - Only consider x in [x_min, x_max] so trailing-edge closure doesn't force min thickness ~0.
      - Bin x and compute thickness per bin = max(y) - min(y)
      - Return minimum thickness across bins

    This is a fast "sanity" constraint, not a perfect CAD thickness computation.
    """
    c = np.asarray(coords, dtype=float)
    if c.ndim != 2 or c.shape[1] != 2:
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0

    x = c[:, 0]
    y = c[:, 1]

    # ignore LE/TE edge regions
    m = (x >= x_min) & (x <= x_max) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 10:
        return 0.0

    bins = np.linspace(x_min, x_max, n_bins + 1)
    t_vals = []
    for i in range(n_bins):
        mm = (x >= bins[i]) & (x < bins[i + 1])
        if np.any(mm):
            t_vals.append(float(np.max(y[mm])) - float(np.min(y[mm])))

    if not t_vals:
        return 0.0

    t = np.asarray(t_vals, dtype=float)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return 0.0

    return float(np.min(t))


def geometry_penalty(coords: np.ndarray, min_thickness: float) -> tuple[float, dict]:
    """
    Penalize if estimated min thickness < min_thickness.
    """
    t_min = _min_thickness_estimate(coords)
    pen = relu(float(min_thickness) - float(t_min))
    return float(pen), {"min_thickness_est": float(t_min)}


def recommend_min_thickness(
    decoded_coords_list: list[np.ndarray],
    *,
    mode: str = "min",
    percentile: float = 5.0,
) -> float:
    """
    Action item: "Check min of all airfoils, use that min for geometry penalty"

    Notes for the meeting:
      * mode="min" follows the action item literally.
      * Sometimes one corrupted / self-intersecting foil makes the true min ~0.
        If that happens, switch to mode="percentile" with a small percentile
        (e.g., 1-5%) for a more robust threshold.

    Usage:
      - decode a batch of foils (or load from dataset)
      - call recommend_min_thickness(...)
      - use the returned value as min_thickness in NOM
    """
    vals = []
    for coords in decoded_coords_list:
        vals.append(_min_thickness_estimate(coords))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0

    mode = str(mode).lower().strip()
    if mode == "min":
        return float(np.min(vals))
    if mode in {"pct", "percentile"}:
        return float(np.percentile(vals, percentile))
    raise ValueError(f"Unknown mode={mode!r}. Use 'min' or 'percentile'.")


# ---------- CL bounds ----------

def cl_bounds_penalty(CL: float, cl_min: float | None = None, cl_max: float | None = None) -> float:
    """
    Penalize CL outside the target window.
    """
    CL = float(CL)
    if not np.isfinite(CL):
        return float("inf")

    pen = 0.0
    if cl_min is not None:
        pen += relu(float(cl_min) - CL)
    if cl_max is not None:
        pen += relu(CL - float(cl_max))
    return float(pen)


# ---------- Total penalty ----------

def total_penalty(
    latent_vec: np.ndarray,
    coords: np.ndarray,
    CL: float | None,
    lat_lo: np.ndarray,
    lat_hi: np.ndarray,
    lam_bounds: float = 1.0,
    lam_geom: float = 5.0,
    lam_cl: float = 10.0,
    min_thickness: float = 0.02,
    cl_min: float | None = None,
    cl_max: float | None = None,
) -> tuple[float, dict]:
    """
    Combine penalties.

    Lambda meanings (action item: comment lambdas):
      lam_bounds: how strongly we keep NOM inside training latent min/max
      lam_geom:   how strongly we avoid ultra-thin/degenerate foils
      lam_cl:     how strongly we enforce lift window

    Total penalty:
      P = lam_bounds*p_bounds + lam_geom*p_geom + lam_cl*p_cl
    """
    # 1) latent bounds
    p_bounds = latent_bounds_penalty(latent_vec, lat_lo, lat_hi)

    # 2) geometry
    p_geom, geom_info = geometry_penalty(coords, min_thickness=min_thickness)

    # 3) CL bounds
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
