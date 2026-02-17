"""
optimization/constraints.py

---------------------------------------------------------------------------
WHAT THIS FILE DOES (plain English):
---------------------------------------------------------------------------
This file contains all the physical and aerodynamic "rules" (constraints)
that a hydrofoil shape must satisfy during NOM optimization.

For each candidate 6-parameter latent vector the optimizer tries, this file
answers four questions:

  1) Are the 6 latent parameters inside the range we saw in our training
     dataset? If the optimizer wanders too far outside, the decoder will
     produce nonsense shapes it was never trained on.
     --> latent_minmax_bounds()   : computes the allowed [lo, hi] box from data
     --> latent_bounds_penalty()  : penalizes any parameter that goes out of range

  2) Does the decoded foil LOOK like a real hydrofoil?
     --> geometry_penalty()
         Checks (in order):
           * Are all coordinates finite and in a valid range?
           * Is the upper surface actually ABOVE the lower surface everywhere?
             (if not, the foil is physically impossible -- hard reject)
           * Is the foil thick enough to be structurally realistic?
             (uses the MINIMUM thickness found across ALL training airfoils)
           * Is the foil not absurdly fat/thick?
           * Is the camber (mean curvature of the foil) within a normal range?
           * Do the leading edge and trailing edge close up properly?

  3) Is the CL (lift coefficient) inside the designer-specified operating window?
     --> cl_bounds_penalty()
         We need CL >= cl_min to generate enough lift to fly the boat.
         We need CL <= cl_max to avoid cavitation at operating speed.

  4) A master function that combines ALL three penalties above into one number.
     --> total_penalty()
         formula: lam_bounds * p_bounds + lam_geom * p_geom + lam_cl * p_cl
         The lambda weights (lam_*) control how strongly each constraint is
         enforced relative to the main CD/CL objective.

---------------------------------------------------------------------------
HARD vs SOFT PENALTIES:
---------------------------------------------------------------------------
  HARD penalty  --> returns float("inf")
    The NOM driver will SKIP this candidate entirely.
    The optimizer can NEVER accept a hard-rejected foil, no matter how good
    its CD/CL looks. Used for physically impossible shapes.

  SOFT penalty  --> returns a positive float >= 0
    The optimizer is discouraged from violating these but CAN trade off
    aerodynamic performance against them. Used for things like TE/LE gaps
    that represent "imperfect" rather than "impossible" shapes.

---------------------------------------------------------------------------
WHY WE REMOVED THE NORMALIZE_CHORD FUNCTION:
---------------------------------------------------------------------------
  The decoder outputs foil coordinates already normalized to chord = 1
  (x in [0, 1]). Normalizing again would be redundant and wrong.

---------------------------------------------------------------------------
WHY MIN_THICKNESS COMES FROM THE DATASET:
---------------------------------------------------------------------------
  Instead of hardcoding a guess for min_thickness, the professor's action
  item was to find the actual minimum thickness across all training airfoils
  and use that as the lower bound. This is computed once in nom_driver.py
  by scanning the dataset and passed in as min_thickness here.

---------------------------------------------------------------------------
NOTE ON THE TRAILING EDGE AND MIN THICKNESS:
---------------------------------------------------------------------------
  Near x=1.0 (the trailing edge), even perfectly valid real airfoils taper
  to near-zero thickness. If we checked min thickness all the way to x=1,
  EVERY foil would fail. So we only check thickness in the INTERIOR of the
  foil: x in [thickness_x_min=0.05, thickness_x_max=0.90].
  The trailing edge is allowed to be thin -- that is physically correct.
---------------------------------------------------------------------------
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Small helper: ReLU = max(0, x)
# ---------------------------------------------------------------------------

def relu(x: float) -> float:
    """
    ReLU stands for "Rectified Linear Unit" -- it simply means max(0, x).

    We use this to convert a constraint violation into a penalty:
      - If the constraint IS satisfied (violation value <= 0) --> penalty = 0
        (no penalty added -- optimizer is not discouraged)
      - If the constraint is VIOLATED (violation value > 0)   --> penalty = violation
        (penalty grows the more badly the constraint is broken)

    Example:
      relu(-5.0) --> 0.0   (thickness is fine, no penalty)
      relu( 0.3) --> 0.3   (thickness is slightly too thin, penalty = 0.3)
      relu( 2.1) --> 2.1   (very thin foil, large penalty)
    """
    return float(max(0.0, x))


# ===========================================================================
# 1) LATENT BOUNDS -- compute allowed range from dataset, then penalize
# ===========================================================================

def latent_minmax_bounds(latents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    PURPOSE:
      Scan the full dataset of latent vectors and compute the per-parameter
      minimum and maximum values. These become the "allowed box" for the
      optimizer: it should not push the latent params outside the range
      seen in training, because the decoder was never trained on those inputs.

    HOW IT CONNECTS TO THE REST OF THE CODE:
      Called once at the start of nom_driver.py:
          all_latents = load_latent_dataset(...)   # shape (N, 6)
          lat_lo, lat_hi = latent_minmax_bounds(all_latents)
      Then lat_lo and lat_hi are passed into latent_bounds_penalty() and
      into total_penalty() every iteration.

    INPUTS:
      latents -- numpy array of shape (N, 6)
                 N rows = one row per airfoil in the training dataset
                 6 cols = p1, p2, p3, p4, p5, p6 (the 6 latent parameters)

    OUTPUTS:
      lo -- shape (6,): the minimum value each parameter is allowed to have
      hi -- shape (6,): the maximum value each parameter is allowed to have

    NOTE on naming -- lo and hi:
      ACTION ITEM (meeting): "change lo and hi (we kept lo and hi to represent
      the upper and lower bounds of the LATENT PARAMETER RANGE, not airfoil surfaces)"
      lo = lower bound of the allowed range  (the MINIMUM value each param can be)
      hi = upper bound of the allowed range  (the MAXIMUM value each param can be)

      Example: if p3 ranged from -0.09 to 3.32 across training data,
        lo[2] = -0.09   (p3 cannot go below this)
        hi[2] =  3.32   (p3 cannot go above this)

      These have NOTHING to do with the upper/lower surfaces of the airfoil.
      They are purely the allowed numeric range for each of the 6 latent parameters.
    """
    Z = np.asarray(latents, dtype=float)

    if Z.ndim != 2 or Z.shape[1] != 6:
        raise ValueError(f"Expected latents shape (N, 6), got {Z.shape}")

    lo = np.nanmin(Z, axis=0)   # shape (6,): minimum of each column across all foils
    hi = np.nanmax(Z, axis=0)   # shape (6,): maximum of each column across all foils

    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("Latent bounds contain NaN/Inf -- check your CSV file.")

    return lo.astype(float), hi.astype(float)


def latent_bounds_penalty(latent_vec: np.ndarray,
                          lo: np.ndarray,
                          hi: np.ndarray) -> float:
    """
    PURPOSE:
      Given a candidate latent vector z = [p1, p2, p3, p4, p5, p6],
      return a penalty that is 0 if all params are inside [lo, hi],
      or positive if any param goes out of that range.

    HOW IT WORKS (for each of the 6 dimensions):
      - If z[i] < lo[i]:   penalize by (lo[i] - z[i])  -- "too far below lower bound"
      - If z[i] > hi[i]:   penalize by (z[i] - hi[i])  -- "too far above upper bound"
      - If lo[i] <= z[i] <= hi[i]:  no penalty (0)

    HOW IT CONNECTS TO THE REST OF THE CODE:
      Called inside total_penalty() with the candidate latent and the bounds
      computed by latent_minmax_bounds(). The result is scaled by lam_bounds
      before being added to the total penalty score.

    INPUTS:
      latent_vec -- shape (6,): the candidate latent params being tested
      lo         -- shape (6,): lower bounds (from latent_minmax_bounds)
      hi         -- shape (6,): upper bounds (from latent_minmax_bounds)

    OUTPUT:
      A single float >= 0. Zero means all params are in-bounds.
    """
    z  = np.asarray(latent_vec, dtype=float).reshape(-1)
    lo = np.asarray(lo,         dtype=float).reshape(-1)
    hi = np.asarray(hi,         dtype=float).reshape(-1)

    if z.shape != lo.shape or z.shape != hi.shape:
        raise ValueError(
            f"Shape mismatch: latent_vec={z.shape}, lo={lo.shape}, hi={hi.shape}"
        )

    below = np.maximum(lo - z, 0.0)   # how far each param is BELOW its lower bound
    above = np.maximum(z - hi, 0.0)   # how far each param is ABOVE its upper bound

    # Sum all violations across all 6 dimensions
    return float(np.sum(below + above))


# ===========================================================================
# 2) GEOMETRY PENALTY -- does the foil look physically real?
# ===========================================================================

def _split_upper_lower(coords: np.ndarray,
                       *,
                       n_points: int
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    PURPOSE:
      Split the 80x2 coords array into the upper and lower surfaces,
      and return BOTH going in the same x-direction (0 -> 1, LE to TE).
      This is needed so we can compare y_upper(x) vs y_lower(x) at the
      same x positions to compute thickness and camber.

    HOW COORDS ARE STORED (talarai_pipeline.py convention):
      coords[0 : n_points]          = UPPER surface, stored TE -> LE (x: 1 -> 0)
      coords[n_points : 2*n_points] = LOWER surface, stored LE -> TE (x: 0 -> 1)

      The upper surface is stored "backwards" because NeuralFoil needs the
      coordinates as a closed loop: upper TE->LE, then lower LE->TE.

    WHAT WE RETURN:
      upper_01 -- upper surface re-ordered so x goes 0 -> 1 (LE to TE)
      lower_01 -- lower surface already going 0 -> 1 (LE to TE)

    HOW IT CONNECTS TO THE REST OF THE CODE:
      Called at the start of geometry_penalty() so that _interp_profiles()
      can interpolate both surfaces onto the same x grid.
    """
    c = np.asarray(coords, dtype=float)

    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError("coords must be shape (N, 2) -- N rows of [x, y]")
    if c.shape[0] < 2 * n_points:
        raise ValueError(
            f"coords has {c.shape[0]} rows, expected at least {2 * n_points}"
        )

    upper_te_to_le = c[:n_points]            # rows 0..39: x goes 1 -> 0 (TE to LE)
    lower_le_to_te = c[n_points:2*n_points]  # rows 40..79: x goes 0 -> 1 (LE to TE)

    # Flip the upper surface so it also goes LE -> TE (x: 0 -> 1)
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
      Interpolate both surfaces onto the SAME evenly-spaced x grid so we
      can compute thickness and camber at every x location.

      Without this step, upper_01 and lower_01 might have their 40 points
      at slightly different x positions, making subtraction meaningless.

    HOW IT CONNECTS TO THE REST OF THE CODE:
      Called inside geometry_penalty() after _split_upper_lower().
      Returns xg, yu_g, yl_g which are then used to compute:
          thickness(x) = yu_g(x) - yl_g(x)
          camber(x)    = 0.5 * (yu_g(x) + yl_g(x))

    INPUTS:
      upper_01, lower_01 -- both going LE->TE (x: 0->1), shape (40, 2)
      n_bins             -- number of points in the shared x grid (default 120)

    OUTPUTS:
      xg    -- shared x grid, shape (n_bins,), values from 0.0 to 1.0
      yu_g  -- upper surface y values at each xg point
      yl_g  -- lower surface y values at each xg point
    """
    xu, yu = upper_01[:, 0], upper_01[:, 1]
    xl, yl = lower_01[:, 0], lower_01[:, 1]

    xg   = np.linspace(0.0, 1.0, int(n_bins)).astype(float)
    yu_g = np.interp(xg, xu, yu)   # interpolate upper y onto shared grid
    yl_g = np.interp(xg, xl, yl)   # interpolate lower y onto shared grid

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
      Check whether a decoded foil shape looks physically realistic.
      Returns a penalty score (0 = perfect, >0 = some violation, inf = impossible).

    HOW IT CONNECTS TO THE REST OF THE CODE:
      Called inside total_penalty(), which is called every iteration of
      the NOM loop in nom_driver.py. If this returns inf, the NOM driver
      skips that candidate entirely and tries a new one.

    WHAT WE CHECK AND WHY:

      [HARD REJECTS -- return float("inf") immediately]
      These mean the shape is physically impossible. The optimizer can NEVER
      keep this foil no matter how good its CD/CL looks.

        - Non-finite coords: the decoder produced NaN or Inf (broken input)
        - x out of range:    foil extends outside [0, 1] chord -- not normalized
        - y too large:       decoder produced a wildly exaggerated shape
        - Surface crossing:  upper surface went below lower surface (thickness < 0)
                             A foil where upper and lower cross is impossible to build.
        - Too thin:          min_thickness < dataset minimum -- physically impossible
                             to manufacture; also causes NeuralFoil to fail.
                             NOTE: only checked in the INTERIOR (x_min to x_max)
                             because the trailing edge is ALLOWED to taper to thin.
        - Too thick:         max_thickness exceeded -- "blob" shape, not a real foil
        - Excess camber:     foil is too curved -- not seen in any real airfoil database

      [SOFT PENALTIES -- return positive float]
      These mean the shape is "imperfect" but potentially usable.
      The optimizer tries to minimize these but is not absolutely blocked.

        - TE gap: trailing edge doesn't close cleanly
                  (upper and lower surface endpoints are too far apart)
        - LE gap: leading edge doesn't close cleanly
                  (same issue at the front of the foil)

    NOTE on thickness_x_max = 0.90 (not 0.95 or 1.0):
      Real airfoils taper sharply near the trailing edge (x > 0.90).
      Even a perfect NACA foil would show thickness < 0.02 at x=0.95.
      So we only enforce minimum thickness in x in [0.05, 0.90] -- the
      true "structural" region of the foil. The trailing edge taper is fine.

    INPUTS:
      coords         -- shape (80, 2): the decoded foil coordinates
                        (in talarai_pipeline.py format: upper TE->LE, lower LE->TE)
      n_points       -- 40 points per surface (must match pipeline)
      min_thickness  -- minimum allowed interior thickness (from dataset minimum)
      max_thickness  -- maximum allowed thickness (prevents blob shapes)
      thickness_x_min/max -- interior x range where thickness is enforced
      camber_max_abs -- maximum allowed absolute camber (mean camber line deviation)
      te_gap_max     -- max allowed trailing edge gap (soft penalty threshold)
      le_gap_max     -- max allowed leading edge gap (soft penalty threshold)
      max_abs_y      -- hard upper bound on |y| to catch decoder explosions
      x_tol          -- tolerance for x being slightly outside [0, 1]
      profile_bins   -- how many x points to use when interpolating thickness

    OUTPUTS:
      (penalty, info_dict)
        penalty  -- float: 0 = all good, >0 = soft violations, inf = hard reject
        info_dict-- dict of diagnostic values (thickness, camber, gaps, etc.)
    """
    c = np.asarray(coords, dtype=float)

    # --- Sanity check: must be a valid 2D array with 2 columns ---
    if c.ndim != 2 or c.shape[1] != 2 or not np.all(np.isfinite(c)):
        return float("inf"), {"reason": "coords_invalid: not a finite (N,2) array"}

    # --- HARD REJECT: x out of normalized range ---
    # All x values must be in [0 - x_tol, 1 + x_tol]. If not, the foil is
    # not chord-normalized and the decoder produced garbage.
    xmin = float(np.min(c[:, 0]))
    xmax = float(np.max(c[:, 0]))
    if xmin < (0.0 - x_tol) or xmax > (1.0 + x_tol):
        return float("inf"), {
            "reason": f"HARD_REJECT x_out_of_range: xmin={xmin:.4f} xmax={xmax:.4f}"
        }

    # --- HARD REJECT: y values too large ---
    # If any |y| > max_abs_y, the decoder has produced an unrealistic blob.
    maxy = float(np.max(np.abs(c[:, 1])))
    if maxy > float(max_abs_y):
        return float("inf"), {
            "reason": f"HARD_REJECT y_too_large: max|y|={maxy:.4f} > {max_abs_y}"
        }

    # --- Split into upper and lower surfaces (both going LE -> TE, x: 0->1) ---
    try:
        upper_01, lower_01 = _split_upper_lower(c, n_points=n_points)
    except Exception as e:
        return float("inf"), {"reason": f"split_failed: {e}"}

    # --- SOFT: Trailing edge (TE) closure gap ---
    # At x=1 (trailing edge), upper and lower should meet (or nearly meet).
    # p_te = how much the TE gap exceeds the allowed maximum.
    te_gap = float(np.linalg.norm(upper_01[-1] - lower_01[-1]))
    p_te   = relu(te_gap - float(te_gap_max))

    # --- SOFT: Leading edge (LE) closure gap ---
    # At x=0 (leading edge), the two surfaces should also nearly meet.
    le_gap = float(np.linalg.norm(upper_01[0] - lower_01[0]))
    p_le   = relu(le_gap - float(le_gap_max))

    # --- Interpolate both surfaces onto a shared x grid ---
    # This lets us compute thickness(x) and camber(x) at the same x values.
    xg, yu, yl = _interp_profiles(upper_01, lower_01, n_bins=int(profile_bins))

    # thickness(x) = y_upper - y_lower  (must be >= 0 everywhere for valid foil)
    # camber(x)    = midpoint between surfaces = mean camber line deviation from zero
    thickness = yu - yl
    camber    = 0.5 * (yu + yl)

    # --- HARD REJECT: Surface crossing ANYWHERE (including LE and TE) ---
    #
    # WHY THIS MUST COME BEFORE THE INTERIOR MASK:
    #   The interior mask (below) only checks x in [0.05, 0.90] for
    #   min/max thickness. This is intentional -- TE naturally tapers thin.
    #   BUT a surface crossing (lower y > upper y) is ALWAYS physically
    #   impossible regardless of where it occurs. A foil with lower above
    #   upper at x=0 is not a foil at all.
    #
    #   Without this check, a foil where the surfaces just barely cross at
    #   the leading edge point (e.g. lower[0].y > upper[0].y by 0.0001)
    #   passes the interior mask check entirely and produces a pathological
    #   shape with a sharp kink at the nose.
    #
    #   We use a small tolerance (-1e-4) to avoid rejecting numerically
    #   near-perfect foils where floating-point gives thickness = -0.000001.
    if np.any(thickness < -1e-4):
        min_t_full = float(np.min(thickness))
        x_cross    = float(xg[np.argmin(thickness)])
        return float("inf"), {
            "reason": (
                f"HARD_REJECT surface_crossing: min_thickness={min_t_full:.6f} "
                f"at x={x_cross:.4f} (checked full range, including LE/TE)"
            ),
            "min_thickness_full": min_t_full,
            "crossing_x": x_cross,
        }

    # --- Interior mask: only check thickness/camber in x = [x_min, x_max] ---
    # We deliberately EXCLUDE near the leading edge (x < 0.05) and near the
    # trailing edge (x > 0.90) because:
    #   - LE: foil naturally sharpens to near-zero thickness at x=0
    #   - TE: foil naturally tapers to thin at x > 0.90 -- this is CORRECT
    #         physics, not a violation. The trailing edge CAN be the minimum
    #         thickness point on a real foil, which is expected and allowed.
    m_int = (xg >= float(thickness_x_min)) & (xg <= float(thickness_x_max))
    if not np.any(m_int):
        return float("inf"), {"reason": "interior_mask_empty: check x_min/x_max"}

    t_int   = thickness[m_int]
    c_int   = camber[m_int]

    min_t   = float(np.min(t_int))
    max_t   = float(np.max(t_int))
    max_cam = float(np.max(np.abs(c_int)))

    # --- HARD REJECT: Interior surface crossing (redundant safety check) ---
    # The full-range crossing check above already catches any crossing.
    # This check is kept as a belt-and-suspenders guard for the interior
    # region specifically, in case floating point produces a small negative
    # value that slipped past the -1e-4 tolerance above.
    if min_t < 0.0:
        return float("inf"), {
            "reason": f"HARD_REJECT surface_crossing_interior: min_t={min_t:.5f} < 0",
            "min_thickness_int": min_t,
            "max_thickness_int": max_t,
        }

    # --- HARD REJECT 2: Too thin ---
    # min_thickness is set from the actual minimum thickness observed across
    # ALL training airfoils (computed in nom_driver.py from the dataset).
    # If the foil is thinner than any real foil we trained on, it is either
    # structurally impossible or outside the decoder's learned space.
    if min_t < float(min_thickness):
        return float("inf"), {
            "reason": f"HARD_REJECT too_thin: min_t={min_t:.5f} < {min_thickness}",
            "min_thickness_int": min_t,
            "max_thickness_int": max_t,
        }

    # --- HARD REJECT 3: Too thick ---
    # Prevents the optimizer from exploiting unrealistic "blob" shapes that
    # happen to score well on CD/CL even though no real foil looks like that.
    if max_t > float(max_thickness):
        return float("inf"), {
            "reason": f"HARD_REJECT too_thick: max_t={max_t:.5f} > {max_thickness}",
            "min_thickness_int": min_t,
            "max_thickness_int": max_t,
        }

    # --- HARD REJECT 4: Excessive camber ---
    # camber is the y-value of the mean camber line (midpoint between surfaces).
    # Real foils have modest camber. If this is extreme, the foil is unrealistic.
    if max_cam > float(camber_max_abs):
        return float("inf"), {
            "reason": f"HARD_REJECT camber: max_cam={max_cam:.5f} > {camber_max_abs}",
            "min_thickness_int": min_t,
            "max_thickness_int": max_t,
            "max_abs_camber_int": max_cam,
        }

    # --- All hard checks passed: sum the soft penalties ---
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
        "p_cross":  0.0,    # passed hard check (no surface crossing)
        "p_tmin":   0.0,    # passed hard check (not too thin)
        "p_tmax":   0.0,    # passed hard check (not too thick)
        "p_camber": 0.0,    # passed hard check (camber in range)
        "p_te":     float(p_te),
        "p_le":     float(p_le),
        "p_y":      0.0,
        "p_x":      0.0,
    }
    return pen, info


# ===========================================================================
# 3) CL WINDOW CONSTRAINT -- is lift coefficient in the right range?
# ===========================================================================

def cl_bounds_penalty(CL: float,
                      *,
                      cl_min: float | None = None,
                      cl_max: float | None = None) -> float:
    """
    PURPOSE:
      Penalize if CL (lift coefficient) falls outside the designer's
      required operating window [cl_min, cl_max].

    WHY WE NEED THIS:
      - CL >= cl_min: the foil must generate ENOUGH lift to fly the boat
      - CL <= cl_max: too much lift risks cavitation at the operating speed,
                      and can indicate an unrealistically high-lift shape

    HOW IT CONNECTS TO THE REST OF THE CODE:
      Called inside total_penalty() if cl_min or cl_max is not None.
      CL comes from NeuralFoil after evaluating the decoded foil shape.
      The result is scaled by lam_cl before being added to the total penalty.

    INPUTS:
      CL     -- float: lift coefficient from NeuralFoil
      cl_min -- float or None: minimum required CL (set to None to disable)
      cl_max -- float or None: maximum allowed CL  (set to None to disable)

    OUTPUT:
      penalty -- float >= 0. Zero if CL is inside the [cl_min, cl_max] window.
    """
    CL = float(CL)
    if not np.isfinite(CL):
        return float("inf")   # NeuralFoil returned garbage -- hard reject this too

    pen = 0.0
    if cl_min is not None:
        # Penalize if CL is below the minimum required lift
        pen += relu(float(cl_min) - CL)
    if cl_max is not None:
        # Penalize if CL is above the maximum safe lift
        pen += relu(CL - float(cl_max))
    return float(pen)


# ===========================================================================
# 4) COMBINED PENALTY -- used by the NOM loop in nom_driver.py
# ===========================================================================

def total_penalty(*,
                  latent_vec: np.ndarray,
                  coords: np.ndarray,
                  CL: float | None,
                  lat_lo: np.ndarray,
                  lat_hi: np.ndarray,

                  # ---------------------------------------------------------------
                  # ACTION ITEM (meeting): "Comment the lambdas to what they mean"
                  #
                  # Lambda (lam_*) weights are MULTIPLIERS on each penalty term.
                  # Think of them as: "how bad is violating this constraint
                  # compared to having a slightly worse CD/CL?"
                  #
                  # The total score each iteration is:
                  #   score = CD/CL  +  lam_bounds * p_bounds
                  #                  +  lam_geom   * p_geom
                  #                  +  lam_cl     * p_cl
                  #
                  # A higher lambda = that constraint matters MORE to the optimizer.
                  # ---------------------------------------------------------------

                  # lam_bounds = 1.0
                  # Penalty weight for the latent parameters going OUTSIDE the range
                  # we saw in training data. If the optimizer tries p3=5.0 when the
                  # training data only ever had p3 up to 3.32, the decoder will
                  # produce nonsense shapes it was never taught. We penalize this
                  # mildly (weight=1) -- it is a soft warning, not a hard block.
                  lam_bounds: float = 1.0,

                  # lam_geom = 25.0
                  # Penalty weight for SOFT geometry issues (leading edge and trailing
                  # edge gaps that are slightly too large). Hard geometry violations
                  # (surfaces crossing, too thin, too thick, too much camber) always
                  # return float("inf") regardless of this weight -- they are always
                  # completely rejected. lam_geom only applies to the "almost fine"
                  # cases like TE gap = 0.012 when limit is 0.010.
                  lam_geom: float = 25.0,

                  # lam_cl = 10.0
                  # Penalty weight for CL being OUTSIDE the [cl_min, cl_max] window.
                  # We need CL in a specific range so the foil actually works on the
                  # boat at the design speed. cl_min ensures enough lift to fly;
                  # cl_max prevents cavitation (bubbles forming on the foil surface
                  # which causes violent vibration and loss of lift).
                  # Weight=10 means a CL violation of 0.1 adds 10*0.1=1.0 to the score,
                  # which is much larger than a typical CD/CL of 0.02 -- so the
                  # optimizer strongly prefers to stay in the CL window.
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
      Combine all three penalty types into a single penalty score that
      the NOM driver adds to the CD/CL objective.

    FORMULA (when all hard geometry checks pass):
      total_penalty = (lam_bounds * p_bounds)
                    + (lam_geom   * p_geom  )
                    + (lam_cl     * p_cl    )

    CRITICAL BEHAVIOR:
      If geometry_penalty() returns inf (hard rejection), we immediately
      return inf WITHOUT computing the other penalties. The NOM driver
      checks this inf and skips the candidate completely -- it can NEVER
      become the "best" result, regardless of CD/CL.

    HOW IT CONNECTS TO THE REST OF THE CODE:
      This is the ONLY penalty function called by nom_driver.py.
      It calls latent_bounds_penalty, geometry_penalty, and cl_bounds_penalty
      internally and combines them into one number.

    INPUTS:
      latent_vec  -- shape (6,): candidate latent params being tested
      coords      -- shape (80, 2): decoded foil coordinates
      CL          -- lift coefficient from NeuralFoil (or None to skip CL check)
      lat_lo      -- shape (6,): lower latent bounds from dataset
      lat_hi      -- shape (6,): upper latent bounds from dataset
      lam_bounds  -- penalty weight for latent out-of-bounds
      lam_geom    -- penalty weight for soft geometry violations
      lam_cl      -- penalty weight for CL window violations
      min_thickness, max_thickness, camber_max_abs -- hard geometry limits
      te_gap_max, le_gap_max -- soft geometry limits (trailing/leading edge gap)
      cl_min, cl_max  -- CL operating window (set to None to skip)

    OUTPUTS:
      (total, info_dict)
        total    -- combined penalty float (0 = perfect, >0 = violated, inf = impossible)
        info_dict-- diagnostics from all three penalty functions combined
    """
    # --- Penalty 1: are the latent params inside the trained data range? ---
    # p_bounds = 0 if all 6 params are inside [lat_lo, lat_hi], else > 0
    p_bounds = latent_bounds_penalty(latent_vec, lat_lo, lat_hi)

    # --- Penalty 2: does the foil shape look physically realistic? ---
    # p_geom = inf if hard reject (crossing, too thin/thick, bad camber)
    # p_geom = small float for soft issues (TE/LE gap), or 0 if perfect
    p_geom, geom_info = geometry_penalty(
        coords,
        min_thickness=min_thickness,
        max_thickness=max_thickness,
        camber_max_abs=camber_max_abs,
        te_gap_max=te_gap_max,
        le_gap_max=le_gap_max,
    )

    # CRITICAL: if geometry is a hard reject, stop immediately and return inf.
    # The NOM driver will see this inf and skip the candidate.
    if not np.isfinite(p_geom):
        return float("inf"), {
            **geom_info,
            "p_bounds": float(p_bounds),
            "p_geom":   float("inf"),
            "p_cl":     0.0,
        }

    # --- Penalty 3: is CL in the designer's required operating window? ---
    # p_cl = 0 if CL is inside [cl_min, cl_max], else > 0
    p_cl = 0.0
    if CL is not None:
        p_cl = cl_bounds_penalty(CL, cl_min=cl_min, cl_max=cl_max)

    # Combine all three penalties with their lambda weights
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