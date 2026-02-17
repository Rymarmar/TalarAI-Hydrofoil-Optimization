from __future__ import annotations
import numpy as np

"""
optimization/constraints.py

---------------------------------------------------------------------------
WHAT THIS FILE DOES (plain English):
---------------------------------------------------------------------------
This file contains all the "rules" (constraints) that a foil shape must
follow. During NOM optimization, we try many candidate 6-parameter
latent vectors. For each one, we decode it into a foil shape and check:

  1) Are the 6 latent parameters inside the range we saw in our training data?
     --> latent_bounds_penalty()

  2) Does the decoded foil LOOK like a real foil?
     --> geometry_penalty()
         * Is the upper surface actually ABOVE the lower surface?
         * Is the foil thick enough to be structurally real?
         * Is the foil not weirdly thick/fat?
         * Is the camber (curvature) within a normal range?
         * Do the leading edge and trailing edge close up properly?

  3) Is the CL (lift coefficient) within a designer-specified window?
     --> cl_bounds_penalty()

  4) One master function that adds all the above together with weights.
     --> total_penalty()

Each penalty is either:
  - HARD: returns float("inf") immediately for physically impossible shapes.
    The optimizer will skip these entirely -- it can NEVER trade aerodynamic
    performance against a hard violation.
  - SOFT: returns a positive float that the optimizer tries to minimize.

---------------------------------------------------------------------------
ROOT CAUSE OF THE "BELLY-DIVE" SHAPE -- AND THE FIX:
---------------------------------------------------------------------------
PROBLEM: When all constraints were SOFT, the optimizer could accept an
invalid crossing foil (tmin < 0) if its CD/CL was good enough to beat the
penalty score. The terminal output showed every "NEW BEST" had tmin < 0,
meaning the optimizer NEVER found a valid shape in 800 iterations.

FIX: Surface crossing, min/max thickness, and camber violations are now
HARD REJECTS (return float("inf")). The optimizer cannot keep a shape
that fails these checks, no matter how good its aerodynamics look.
---------------------------------------------------------------------------
"""


# ---------------------------------------------------------------------------
# Small helper: ReLU = max(0, x)
# ---------------------------------------------------------------------------
def relu(x: float) -> float:
    """
    ReLU = "Rectified Linear Unit" -- just means max(0, x).

    We use this everywhere because:
      - If a constraint IS satisfied  (violation <= 0) --> penalty = 0
      - If a constraint is VIOLATED   (violation  > 0) --> penalty = violation

    Example:
      relu(-5.0)  -->  0.0   (no penalty, thickness is fine)
      relu( 0.3)  -->  0.3   (penalty of 0.3, foil is slightly too thin)
    """
    return float(max(0.0, x))


# ===========================================================================
# 1) LATENT BOUNDS CONSTRAINT
# ===========================================================================

def latent_minmax_bounds(latents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    PURPOSE:
      Read the entire dataset of latent vectors (one row per foil, 6 columns
      per row = p1..p6) and return the per-dimension MIN and MAX.

    WHY:
      The decoder was trained on foils whose latent params fall in a certain
      range. If the optimizer wanders outside that range, the decoder will
      produce nonsense shapes (it has never seen those inputs).
      So we find the actual min/max from the data and use those as bounds.

    INPUTS:
      latents  -- numpy array of shape (N, 6)
                  N = number of foils in our dataset
                  6 columns = p1, p2, p3, p4, p5, p6

    OUTPUTS:
      lo  -- numpy array of shape (6,): minimum value of each param (lower bound)
      hi  -- numpy array of shape (6,): maximum value of each param (upper bound)
    """
    Z = np.asarray(latents, dtype=float)

    if Z.ndim != 2 or Z.shape[1] != 6:
        raise ValueError(f"Expected latents shape (N,6), got {Z.shape}")

    lo = np.nanmin(Z, axis=0)   # shape (6,): min of each column
    hi = np.nanmax(Z, axis=0)   # shape (6,): max of each column

    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("Latent bounds contain NaN/Inf. Check your CSV file.")

    return lo.astype(float), hi.astype(float)


def latent_bounds_penalty(latent_vec: np.ndarray,
                          lo: np.ndarray,
                          hi: np.ndarray) -> float:
    """
    PURPOSE:
      Check if a candidate latent vector z = [p1..p6] is inside the
      allowed box [lo, hi]. Penalize any dimension that goes out of range.

    HOW IT WORKS (for each dimension i):
      - If z[i] < lo[i]:  penalize by (lo[i] - z[i])   "too low"
      - If z[i] > hi[i]:  penalize by (z[i] - hi[i])   "too high"
      - If lo[i] <= z[i] <= hi[i]:  no penalty (0)

    NAMING:
      lo = lower bound = the MINIMUM value each param is allowed to have
      hi = upper bound = the MAXIMUM value each param is allowed to have
      These come from the actual min/max of our training dataset.

    OUTPUT:
      A single float >= 0. Zero means all params are in-bounds.
    """
    z  = np.asarray(latent_vec, dtype=float).reshape(-1)
    lo = np.asarray(lo,         dtype=float).reshape(-1)
    hi = np.asarray(hi,         dtype=float).reshape(-1)

    if z.shape != lo.shape or z.shape != hi.shape:
        raise ValueError(f"Shape mismatch: z={z.shape}, lo={lo.shape}, hi={hi.shape}")

    below = np.maximum(lo - z, 0.0)   # how far below the lower bound
    above = np.maximum(z - hi, 0.0)   # how far above the upper bound

    return float(np.sum(below + above))


# ===========================================================================
# 2) GEOMETRY PENALTY (the main physics check)
# ===========================================================================

def _split_upper_lower(coords: np.ndarray,
                       *,
                       n_points: int
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    PURPOSE:
      Split the flat coords array (shape 80x2) into the upper and lower
      surfaces, and return both going in the same x direction (0 -> 1,
      i.e. leading edge to trailing edge).

    HOW THE COORDS ARE STORED (our pipeline convention from talarai_pipeline.py):
      coords[0 : n_points]          = UPPER surface, going TE -> LE  (x: 1 -> 0)
      coords[n_points : 2*n_points] = LOWER surface, going LE -> TE  (x: 0 -> 1)

    WHAT WE RETURN:
      upper_01  -- upper surface re-ordered so x goes 0 -> 1
      lower_01  -- lower surface already going 0 -> 1

    WHY WE REVERSE UPPER:
      The upper surface is stored backwards (trailing edge first) because
      NeuralFoil wants a closed loop: TE -> LE (upper) -> TE (lower).
      But for thickness calculations we need both surfaces going the same
      direction so we can compare y values at the same x location.
    """
    c = np.asarray(coords, dtype=float)

    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError("coords must be shape (N, 2)")
    if c.shape[0] < 2 * n_points:
        raise ValueError(f"coords has {c.shape[0]} rows, expected >= {2*n_points}")

    upper_te_to_le = c[:n_points]           # x goes 1 -> 0
    lower_le_to_te = c[n_points:2*n_points] # x goes 0 -> 1

    # Reverse the upper surface so it also goes 0 -> 1
    upper_01 = upper_te_to_le[::-1].copy()
    lower_01 = lower_le_to_te.copy()

    return upper_01, lower_01


def _interp_profiles(upper_01: np.ndarray,
                     lower_01: np.ndarray,
                     *,
                     n_bins: int = 120
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PURPOSE:
      Interpolate both surfaces onto the SAME set of x values.
      This is needed so we can subtract y_lower(x) from y_upper(x)
      at each x location to get thickness.

    OUTPUTS:
      xg    -- shared x grid, shape (n_bins,)
      yu_g  -- upper y values at each xg point
      yl_g  -- lower y values at each xg point
    """
    xu, yu = upper_01[:, 0], upper_01[:, 1]
    xl, yl = lower_01[:, 0], lower_01[:, 1]

    xg   = np.linspace(0.0, 1.0, int(n_bins)).astype(float)
    yu_g = np.interp(xg, xu, yu)
    yl_g = np.interp(xg, xl, yl)

    return xg, yu_g, yl_g


def geometry_penalty(coords: np.ndarray,
                     *,
                     n_points: int = 40,
                     min_thickness: float = 0.02,
                     max_thickness: float = 0.20,
                     thickness_x_min: float = 0.05,
                     thickness_x_max: float = 0.90,
                     camber_max_abs: float = 0.08,
                     te_gap_max: float = 0.01,
                     le_gap_max: float = 0.01,
                     max_abs_y: float = 0.25,
                     x_tol: float = 0.02,
                     profile_bins: int = 120,
                     ) -> tuple[float, dict]:
    """
    PURPOSE:
      Check whether a decoded foil shape looks physically reasonable.

    HARD REJECTS (returns float("inf") immediately -- optimizer SKIPS these):
      - Coords are non-finite or x/y out of range
      - Upper and lower surfaces CROSS (thickness goes negative)
      - Foil too thin (structural impossibility)
      - Foil too thick (cartoon shape)
      - Excessive camber (unrealistic curvature)

    SOFT PENALTIES (returns positive float -- optimizer tries to reduce):
      - TE gap: trailing edge doesn't close cleanly
      - LE gap: leading edge doesn't close cleanly

    WHY HARD vs SOFT:
      Hard rejects for crossing/thickness/camber prevent the optimizer from
      ever keeping an invalid foil as its "best" result -- even if that foil
      has a low CD/CL. This was the root cause of the belly-dive shape.

    thickness_x_max is 0.90 (not 0.95) because near the trailing edge the
    foil tapers sharply and even valid NACA foils drop below 0.02 there.
    Using 0.90 excludes that natural taper region from the min check.
    """
    c = np.asarray(coords, dtype=float)

    # Basic sanity: must be finite 2D array with 2 columns
    if c.ndim != 2 or c.shape[1] != 2 or not np.all(np.isfinite(c)):
        return float("inf"), {"reason": "coords_invalid"}

    # --- x-range sanity: all x values must be in [0-tol, 1+tol] ---
    xmin = float(np.min(c[:, 0]))
    xmax = float(np.max(c[:, 0]))
    if xmin < (0.0 - x_tol) or xmax > (1.0 + x_tol):
        return float("inf"), {
            "reason": f"HARD_REJECT x_out_of_range: xmin={xmin:.4f} xmax={xmax:.4f}"
        }

    # --- y-range sanity: no y value should be wildly large ---
    # HARD REJECT: max|y| > max_abs_y means a nonsense decoder output
    maxy = float(np.max(np.abs(c[:, 1])))
    if maxy > float(max_abs_y):
        return float("inf"), {
            "reason": f"HARD_REJECT y_too_large: max|y|={maxy:.4f} > {max_abs_y}"
        }

    # --- Split into upper and lower surfaces ---
    try:
        upper_01, lower_01 = _split_upper_lower(c, n_points=n_points)
    except Exception as e:
        return float("inf"), {"reason": f"split_failed: {e}"}

    # --- TE and LE closure gaps (SOFT penalties) ---
    te_gap = float(np.linalg.norm(upper_01[-1] - lower_01[-1]))
    p_te   = relu(te_gap - float(te_gap_max))

    le_gap = float(np.linalg.norm(upper_01[0] - lower_01[0]))
    p_le   = relu(le_gap - float(le_gap_max))

    # --- Interpolate onto shared x grid ---
    xg, yu, yl = _interp_profiles(upper_01, lower_01, n_bins=int(profile_bins))

    # thickness(x) = y_upper(x) - y_lower(x) -- must be positive everywhere
    thickness = yu - yl
    # camber(x) = midpoint between the two surfaces (the "mean camber line")
    camber = 0.5 * (yu + yl)

    # --- Interior mask ---
    # Near x=0 (LE) and x=1 (TE) the foil naturally tapers to a thin edge.
    # This is expected and NOT a violation -- we only check the interior.
    # NOTE: thickness_x_max = 0.90 (not 0.95) because valid NACA foils
    # taper below 0.02 near x=0.93-0.95 -- using 0.90 avoids false rejects.
    m_int = (xg >= float(thickness_x_min)) & (xg <= float(thickness_x_max))
    if not np.any(m_int):
        return float("inf"), {"reason": "interior_mask_empty"}

    t_int = thickness[m_int]
    c_int = camber[m_int]

    min_t   = float(np.min(t_int))
    max_t   = float(np.max(t_int))
    max_cam = float(np.max(np.abs(c_int)))

    # ==========================================================================
    # HARD REJECT 1: Surface crossing
    # If min_t < 0, upper surface went below lower surface -- physically
    # impossible. HARD REJECT so the optimizer never keeps this shape.
    # ==========================================================================
    if min_t < 0.0:
        return float("inf"), {
            "reason": f"HARD_REJECT crossing: min_t={min_t:.5f}",
            "min_thickness_int": min_t,
            "max_thickness_int": max_t,
        }

    # ==========================================================================
    # HARD REJECT 2: Too thin
    # A foil thinner than min_thickness is not structurally realistic.
    # ==========================================================================
    if min_t < float(min_thickness):
        return float("inf"), {
            "reason": f"HARD_REJECT too_thin: min_t={min_t:.5f} < {min_thickness}",
            "min_thickness_int": min_t,
            "max_thickness_int": max_t,
        }

    # ==========================================================================
    # HARD REJECT 3: Too thick
    # Prevents optimizer from exploiting cartoon "blob" shapes.
    # ==========================================================================
    if max_t > float(max_thickness):
        return float("inf"), {
            "reason": f"HARD_REJECT too_thick: max_t={max_t:.5f} > {max_thickness}",
            "min_thickness_int": min_t,
            "max_thickness_int": max_t,
        }

    # ==========================================================================
    # HARD REJECT 4: Excessive camber
    # ==========================================================================
    if max_cam > float(camber_max_abs):
        return float("inf"), {
            "reason": f"HARD_REJECT camber: max_cam={max_cam:.5f} > {camber_max_abs}",
            "min_thickness_int": min_t,
            "max_thickness_int": max_t,
            "max_abs_camber_int": max_cam,
        }

    # --- All hard checks passed: only soft TE/LE closure penalties remain ---
    pen = float(p_te + p_le)

    info = {
        "xmin": xmin,
        "xmax": xmax,
        "max_abs_y": maxy,
        "te_gap": te_gap,
        "le_gap": le_gap,
        "min_thickness_int": min_t,
        "max_thickness_int": max_t,
        "max_abs_camber_int": max_cam,
        "p_cross":  0.0,   # passed hard check
        "p_tmin":   0.0,   # passed hard check
        "p_tmax":   0.0,   # passed hard check
        "p_camber": 0.0,   # passed hard check
        "p_te":     float(p_te),
        "p_le":     float(p_le),
        "p_y":      0.0,
        "p_x":      0.0,
    }
    return pen, info


# ===========================================================================
# 3) CL (LIFT COEFFICIENT) WINDOW CONSTRAINT
# ===========================================================================

def cl_bounds_penalty(CL: float,
                      *,
                      cl_min: float | None = None,
                      cl_max: float | None = None) -> float:
    """
    PURPOSE:
      Penalize if the lift coefficient CL is outside the designer-specified
      window [cl_min, cl_max].

    WHY:
      We need at least cl_min to lift the boat, and CL > cl_max risks
      cavitation at our operating speed.

    OUTPUT:
      penalty -- float >= 0. Zero if CL is inside the window.
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


# ===========================================================================
# 4) COMBINED PENALTY (used by the NOM loop in nom_driver.py)
# ===========================================================================

def total_penalty(*,
                  latent_vec: np.ndarray,
                  coords: np.ndarray,
                  CL: float | None,
                  lat_lo: np.ndarray,
                  lat_hi: np.ndarray,

                  # lam_bounds: keeps optimizer inside trained decoder range
                  lam_bounds: float = 1.0,

                  # lam_geom: weight for soft geometry penalties (TE/LE gaps).
                  # NOTE: hard geometry violations (crossing, thick, thin, camber)
                  # return inf regardless of this weight.
                  lam_geom: float = 25.0,

                  # lam_cl: weight for CL window penalty
                  lam_cl: float = 10.0,

                  min_thickness: float = 0.02,
                  max_thickness: float = 0.20,
                  camber_max_abs: float = 0.08,
                  te_gap_max: float = 0.01,
                  le_gap_max: float = 0.01,

                  cl_min: float | None = None,
                  cl_max: float | None = None,
                  ) -> tuple[float, dict]:
    """
    PURPOSE:
      Combine all three penalty types into one total penalty score.

    FORMULA (when geometry passes all hard checks):
      total = lam_bounds * p_bounds
            + lam_geom   * p_geom
            + lam_cl     * p_cl

    KEY: If geometry_penalty returns float("inf") (hard rejection), we
    propagate inf immediately. The NOM driver checks for inf and skips
    the candidate -- it can NEVER become the best result.

    This is the critical fix: previously total_penalty just multiplied
    lam_geom * inf = inf, but the NOM driver wasn't checking for it.
    Now the driver explicitly skips any candidate where total is inf.
    """
    # --- Penalty 1: latent params in-range? ---
    p_bounds = latent_bounds_penalty(latent_vec, lat_lo, lat_hi)

    # --- Penalty 2: does the foil look like a real foil? ---
    p_geom, geom_info = geometry_penalty(
        coords,
        min_thickness=min_thickness,
        max_thickness=max_thickness,
        camber_max_abs=camber_max_abs,
        te_gap_max=te_gap_max,
        le_gap_max=le_gap_max,
    )

    # CRITICAL: propagate hard rejection immediately
    # If geometry returns inf, the driver will skip this candidate
    if not np.isfinite(p_geom):
        return float("inf"), {
            **geom_info,
            "p_bounds": float(p_bounds),
            "p_geom":   float("inf"),
            "p_cl":     0.0,
        }

    # --- Penalty 3: CL in the desired window? ---
    p_cl = 0.0
    if CL is not None:
        p_cl = cl_bounds_penalty(CL, cl_min=cl_min, cl_max=cl_max)

    total = float(lam_bounds * p_bounds
                + lam_geom   * p_geom
                + lam_cl     * p_cl)

    info = {
        **geom_info,
        "p_bounds": float(p_bounds),
        "p_geom":   float(p_geom),
        "p_cl":     float(p_cl),
    }
    return total, info