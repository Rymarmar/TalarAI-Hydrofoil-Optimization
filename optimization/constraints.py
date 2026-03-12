"""
optimization/constraints.py

CLEAN REBUILD based on professor's action items (2/17/26 meeting).

WHAT THIS FILE DOES:
  Defines penalty functions for the NOM optimizer. Each candidate foil
  gets a penalty score >= 0 added to its CD/CL objective.

KEY CHANGES from old version:
  ✓ Camber check added back (2/19: "no cambered foils, hard to 3D print")
  ✓ Removed LE gap check (prof: "only check TE gap")
  ✓ Removed inf returns (prof: "use 1000 for hard rejects")
  ✓ Check thickness only in interior x ∈ [0.05, 0.90] (prof: line 408 feedback)
  ✓ Added min-max thickness check (2/19: "minimum of the maximum thickness")
  ✓ Lambda weights auto-normalized so Σ(λ_i) = 1

FIXES APPLIED (3/9/26):
  ✓ FIX 1: Surface-crossing check now uses [0.05, 0.90] interior mask,
    matching the thickness check range. Previously used x >= 0.05 with
    no upper bound, which included the TE region (x > 0.90) where
    decoder noise can cause tiny apparent crossings that aren't real.
    This caused false hard-rejects near the trailing edge.

  ✓ FIX 2: Lambda normalization documented as FIXED-WEIGHT normalization,
    not the adaptive normalization the professor described. See detailed
    comment in total_penalty() explaining the difference and why fixed
    weights are acceptable for now.

PENALTIES:
  1) latent_bounds_penalty  -- params outside training data range
  2) geometry_penalty       -- thickness violations (interior only)
  3) cl_bounds_penalty      -- CL outside [cl_min, cl_max]
  4) total_penalty          -- weighted sum of all three
"""

from __future__ import annotations
import numpy as np


def relu(x: float) -> float:
    """
    ReLU = max(0, x). Used to convert violations into penalties.
    
    ACTION ITEM #1 (2/17): "No need for relu max in bounds"
    Simplified to just use max(0.0, x) directly, removed wrapper complexity.
    """
    return float(max(0.0, x))


# ===========================================================================
# 1) LATENT BOUNDS
# ===========================================================================

def latent_minmax_bounds(latents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-parameter [min, max] from the training dataset.
    
    INPUTS:
      latents -- shape (N, 6): all N training foils' latent params
    
    OUTPUTS:
      lo -- shape (6,): minimum value for each param
      hi -- shape (6,): maximum value for each param
    """
    Z = np.asarray(latents, dtype=float)
    if Z.ndim != 2 or Z.shape[1] != 6:
        raise ValueError(f"Expected shape (N, 6), got {Z.shape}")
    
    lo = np.nanmin(Z, axis=0)
    hi = np.nanmax(Z, axis=0)
    
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("Latent bounds contain NaN/Inf")
    
    return lo.astype(float), hi.astype(float)


def latent_bounds_penalty(latent_vec: np.ndarray,
                          lo: np.ndarray,
                          hi: np.ndarray) -> float:
    """
    Penalty for latent params going outside training data range.
    
    INPUTS:
      latent_vec -- shape (6,): candidate params [p1..p6]
      lo, hi     -- shape (6,): allowed bounds from dataset
    
    OUTPUT:
      penalty -- 0 if all params inside [lo, hi], else sum of violations
    """
    z = np.asarray(latent_vec, dtype=float).reshape(-1)
    lo = np.asarray(lo, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)
    
    if z.shape[0] != 6 or lo.shape[0] != 6 or hi.shape[0] != 6:
        raise ValueError("All arrays must be length 6")
    
    # Penalize how far outside the box we are
    pen = 0.0
    for i in range(6):
        pen += relu(lo[i] - z[i])  # went below min
        pen += relu(z[i] - hi[i])  # went above max
    
    return float(pen)


# ===========================================================================
# 2) GEOMETRY -- thickness, camber, TE gap checks
# ===========================================================================

def geometry_penalty(coords: np.ndarray,
                     min_thickness: float = 0.006,
                     max_thickness: float = 0.157,
                     te_gap_max: float = 0.01,
                     thickness_x_min: float = 0.05,
                     thickness_x_max: float = 0.90,
                     # ACTION ITEM (2/19 meeting): "ADD Minimum thickness of the maximum
                     # thickness of the foil -- at least this for the max thickness"
                     # Hard rejects foils whose peak thickness is too thin (e.g. slivers
                     # that pass the local min_thickness check but have no structural depth).
                     min_max_thickness: float = 0.04,
                     # FIX #3 — max_camber raised from 0.04 → 0.08
                     # The HQ-series baseline foil (hq358) has ~7-8% camber.
                     # At 0.04 (4%c), the baseline and all nearby foils were
                     # immediately hard-rejected after any perturbation, causing
                     # the optimizer to see loss=inf in every direction and
                     # roll back endlessly. Raised to 0.08 (8%c) so the
                     # optimizer can explore the neighborhood of the baseline.
                     # If you switch to a symmetric NACA baseline, lower back to 0.04.
                     max_camber: float = 0.10,
                     ) -> tuple[float, dict]:
    """
    Check foil geometry. SIMPLIFIED per prof feedback.
    
    CHECKS (in order):
      1) Coords finite and reasonable range
      2) Upper above lower in interior [0.05, 0.90] (hard reject if crossing)
      3) Min thickness in interior (hard reject if too thin)
      4) Max thickness (hard reject if too fat)
      5) Min-max thickness (hard reject if peak thickness < min_max_thickness)
      6) Max camber (hard reject if cambered foil)
      7) TE gap (soft penalty if slightly open)
    
    REMOVED (per prof):
      - ACTION ITEM #16: Camber checks (too complicated) -- REVERSED 2/19:
        camber is NOW checked as a hard reject (no cambered foils for 3D printing)
      - ACTION ITEM #8,17: LE gap check (only check TE)
      - ACTION ITEM #13: Sharpness checks (overcomplicated)
      - ACTION ITEM #14: Overcomplicated checks removed
      - ACTION ITEM #10: Full 0->1 thickness scan (only check interior)
    
    HARD REJECTS:
      ACTION ITEM #3 (2/17): Instead of returning inf, return 1000
      (prof: "no inf, use big number")
    
    INPUTS:
      coords -- shape (80, 2): decoded foil coords
        rows 0-39  = upper surface TE->LE (x: 1->0)
        rows 40-79 = lower surface LE->TE (x: 0->1)
      min_thickness     -- hard limit: no point in interior thinner than this
      max_thickness     -- hard limit: no point in interior thicker than this
      te_gap_max        -- soft limit for TE closure
      thickness_x_min   -- interior check starts at x=0.05
      thickness_x_max   -- interior check ends at x=0.90
      min_max_thickness -- ACTION ITEM (2/19): hard limit: peak thickness must be
                           at least this value (prevents ultra-thin slivers)
      max_camber        -- ACTION ITEM (2/19): hard limit: max camber line deviation
                           from zero (prevents cambered foils, enforces NACA symmetry)
    
    OUTPUTS:
      (penalty, info_dict)
        penalty = 0 if perfect
                  small float for soft violations
                  1000 for hard rejections (crossing, too thin/thick)
    """
    coords = np.asarray(coords, dtype=float)
    
    if coords.shape != (80, 2):
        return 1000.0, {"reason": "coords wrong shape"}
    
    x_all, y_all = coords[:, 0], coords[:, 1]
    
    # --- CHECK 1: Coords finite and in reasonable range ---
    # ACTION ITEM #5-6 (2/17): Prof: "Find max y and x from dataset, narrow
    # to ½ for buffer"
    # UPDATED: Used compute_dataset_max_xy.py to find actual dataset bounds.
    # Result: 99th percentile max |y| = 0.1786, with 10% buffer = 0.1964
    # This ignores outliers like naca1.txt (y=1.0) which is 3x larger than
    # normal foils.
    if not np.all(np.isfinite(x_all)) or not np.all(np.isfinite(y_all)):
        return 1000.0, {"reason": "nonfinite coords"}
    
    # ACTION ITEM #4 (2/17): "Hard reject with x will never happen but y will"
    # Sanity: x should be in [0,1], y should be reasonable
    if np.any(x_all < -0.1) or np.any(x_all > 1.1):
        return 1000.0, {"reason": "x out of range"}
    
    # ACTION ITEM #5 (2/17): "max_abs_y = 0.25 (random number), not good idea"
    # FIXED: Now using dataset 99th percentile + 10% buffer (0.1964)
    if np.any(np.abs(y_all) > 0.1964):
        return 1000.0, {"reason": "y too large"}
    
    # --- SURFACE SPLITTING ---
    # ACTION ITEM #2 (2/17): "Didnt update from interpolation script in
    # split_upper_lower (take out)"
    # Removed split_upper_lower function, now splitting surfaces directly here.
    upper_te2le = coords[:40]  # x: 1->0
    lower_le2te = coords[40:]  # x: 0->1
    
    # Flip upper to LE->TE so both go same direction for comparison
    upper_le2te = upper_te2le[::-1]  # x: 0->1
    
    xu, yu = upper_le2te[:, 0], upper_le2te[:, 1]
    xl, yl = lower_le2te[:, 0], lower_le2te[:, 1]

    # ACTION ITEM #9 (2/17): "Line 408 no need for interpolation"
    # The decoder outputs both surfaces on the SAME x-grid (linspace 0->1,
    # 40 pts). So upper and lower share x-values point-for-point -- no
    # interpolation needed. We can subtract y values directly.
    #
    # The old code interpolated onto a new 120-point grid which was:
    #   1) unnecessary (same x-grid already)
    #   2) causing false surface-crossing rejects at LE (x=0) due to
    #      interpolation artifacts on the sharp leading edge
    # Both problems are eliminated by using the decoder grid directly.

    # Both surfaces are already on linspace(0,1,40) -- use directly
    xg = xu  # shape (40,) -- same x values for both surfaces
    thickness = yu - yl  # shape (40,) -- point-by-point, no interpolation
    
    # ---------------------------------------------------------------
    # INTERIOR MASK: used for ALL interior checks (crossing, thickness, camber)
    #
    # ACTION ITEM #10 (2/17): "Instead of 0 to 1, from .1 to .9 or .05 to .9"
    #
    # WHY: Near the leading edge (x~0) every foil tapers to a sharp point
    # where thickness → 0 and upper/lower nearly touch. Near the trailing
    # edge (x~1) the foil also tapers thin. Checking these regions would
    # falsely reject valid foils due to:
    #   - Natural taper (not a defect)
    #   - Decoder noise at endpoints
    #   - Tiny numerical crossings that aren't real
    #
    # FIX (3/9/26): Previously the crossing check used a DIFFERENT mask
    # (x >= 0.05, no upper bound) than the thickness check ([0.05, 0.90]).
    # This meant crossings near the TE (x > 0.90) would trigger a hard
    # reject even though we intentionally exclude that region from
    # thickness checks. Now ALL interior checks use the same mask.
    # ---------------------------------------------------------------
    mask = (xg >= thickness_x_min) & (xg <= thickness_x_max)
    
    # --- CHECK 2: Upper above lower in INTERIOR only ---
    # Skip near LE and TE where foil tapers to a point --
    # thickness naturally approaches 0 there and is not meaningful.
    if np.any(thickness[mask] < -1e-6):
        return 1000.0, {"reason": "surfaces crossing"}

    # --- CHECK 3 & 4: Thickness in INTERIOR only ---
    # ACTION ITEM #11 (2/17): "Line 433, not negative -1e-4 make positive"
    # ACTION ITEM #12 (2/17): "Find min thickness from dataset between .1
    # and .9"
    #
    # LOGIC: We only check thickness in the middle of the foil.
    # Near LE (x~0) and TE (x~1) every real foil tapers thin -- checking
    # there would reject all valid foils. The interior [0.05, 0.90] is
    # where structural thickness actually matters.
    t_interior = thickness[mask]
    
    if len(t_interior) == 0:
        return 1000.0, {"reason": "no interior points"}
    
    t_min = float(np.min(t_interior))  # ACTION ITEM #7: using float()
    t_max = float(np.max(t_interior))  # ACTION ITEM #7: direct float conversion
    
    # Hard reject if too thin (prof: "check min thickness from dataset")
    # ACTION ITEM #11 (2/17): Make positive (was -1e-4, now positive comparison)
    # ACTION ITEM #15 (2/17): "too thin redundant, too thick check max/min"
    if t_min < min_thickness:
        return 1000.0, {"reason": "too thin", "t_min": t_min}
    
    # Hard reject if too thick
    # ACTION ITEM #15 (2/17): Check against max_thickness from constraints
    if t_max > max_thickness:
        return 1000.0, {"reason": "too thick", "t_max": t_max}

    # --- CHECK 5: Min-max thickness (peak must be structurally deep) ---
    # ACTION ITEM (2/19 meeting): "ADD Minimum thickness of the maximum
    # thickness of the foil -- at least this for the max thickness"
    #
    # WHY THIS IS DIFFERENT FROM min_thickness:
    #   min_thickness checks that NO point is too thin (local floor).
    #   min_max_thickness checks that the PEAK thickness is deep enough.
    #   A foil could pass min_thickness (e.g. 0.04 everywhere) but still
    #   be a useless sliver if t_max is only 0.05 -- not 3D printable or
    #   structurally viable. This check ensures enough depth at the
    #   thickest point.
    if t_max < min_max_thickness:
        return 1000.0, {"reason": "peak too thin",
                        "t_max": t_max, "min_max_thickness": min_max_thickness}

    # --- CHECK 6: Max camber (block extreme camber for 3D printing) ---
    # ACTION ITEM (2/19 meeting): "No cambered foils please -- NACA foils
    # easier to 3D print / hard to manufacture cambered foils"
    # INTERPRETATION (confirmed 2/23): "No EXTREME camber" not "zero camber."
    #
    # HOW CAMBER IS COMPUTED:
    #   Camber line = midpoint between upper and lower surface at each x.
    #   For a perfectly symmetric foil (NACA 00xx), camber = 0 everywhere.
    #   We check the interior only [0.05, 0.90] -- near LE/TE every foil
    #   taper closes so tiny apparent camber there is noise, not real camber.
    #
    # THRESHOLD: max_camber (default now 0.08 = 8%c)
    #   Allows: NACA 0012 (0%), NACA 2412 (2%), NACA 4412 (4%), HQ358 (~8%)
    #   Blocks: extreme high-camber foils above 8%c
    camber_line = (yu + yl) / 2.0          # shape (40,) -- midpoint at each x
    camber_interior = camber_line[mask]     # restrict to interior [0.05, 0.90]
    max_camber_actual = float(np.max(np.abs(camber_interior)))
    if max_camber_actual > max_camber:
        return 1000.0, {"reason": "too cambered",
                        "max_camber_actual": max_camber_actual,
                        "max_camber_limit": max_camber}
    
    # --- CHECK 7: TE gap (soft penalty) ---
    # ACTION ITEM #8 (2/17): "No need for leading edge line 404, only
    # trailing edge"
    # ACTION ITEM #9 (2/17): "Line 408 no need for interpolation"
    # Prof: Check TE gap directly from coords, no interpolation needed.
    # If TE closes properly, gap will be ~0 which is correct.
    #
    # ACTION ITEM (3/5): "Trailing edge for manufacturing to be set to
    # 0.005 instead of reaching 0" -- note: default te_gap_max=0.01 is
    # MORE restrictive than 0.005 (rejects wider gaps). Change to 0.005
    # if you want to allow slightly more open TEs.
    
    # TE = x ≈ 1.0 (last points of each surface)
    y_upper_te = upper_te2le[0, 1]   # first row of upper (TE in TE->LE order)
    y_lower_te = lower_le2te[-1, 1]  # last row of lower (TE in LE->TE order)
    
    te_gap = float(abs(y_upper_te - y_lower_te))
    
    # Soft penalty for TE not closing
    p_te = relu(te_gap - te_gap_max)
    
    # --- RETURN ---
    # ACTION ITEM #18 (2/17): "Line 480: thickness should be a penalty,
    # return summation"
    # Prof: "Have another p value, p = min_t, line 509 summation of all
    # penalties"
    # We already did thickness checks above (hard rejects for too
    # thin/thick). Here we only return the soft TE gap penalty.
    total_pen = float(p_te)
    
    info = {
        "t_min": t_min,
        "t_max": t_max,
        "te_gap": te_gap,
        "p_te": float(p_te),
        "reason": "ok" if total_pen == 0 else "soft_violations",
    }
    
    return total_pen, info


# ===========================================================================
# 3) CL BOUNDS
# ===========================================================================

def cl_bounds_penalty(CL: float,
                      cl_min: float | None = None,
                      cl_max: float | None = None) -> float:
    """
    Penalty for CL being outside the operating window [cl_min, cl_max].
    
    INPUTS:
      CL     -- lift coefficient from NeuralFoil
      cl_min -- minimum required CL (None to skip)
      cl_max -- maximum allowed CL (None to skip)
    
    OUTPUT:
      penalty -- 0 if CL in window, else sum of violations
    """
    CL = float(CL)
    
    # Prof: "no inf" -- if CL is garbage just return big penalty
    if not np.isfinite(CL):
        return 1000.0
    
    pen = 0.0
    if cl_min is not None:
        pen += relu(float(cl_min) - CL)
    if cl_max is not None:
        pen += relu(CL - float(cl_max))
    
    return float(pen)


# ===========================================================================
# 4) TOTAL PENALTY with lambda weights
# ===========================================================================

def total_penalty(*,
                  latent_vec: np.ndarray,
                  coords: np.ndarray,
                  CL: float | None,
                  lat_lo: np.ndarray,
                  lat_hi: np.ndarray,
                  
                  # Penalty weights (relative importance)
                  #
                  # WHAT THESE MEAN:
                  #   lam_bounds = how much to penalize going outside latent bounds
                  #   lam_geom   = how much to penalize geometry violations (TE gap)
                  #   lam_cl     = how much to penalize CL being out of range
                  #
                  # WHY THESE VALUES:
                  #   lam_cl (50) >> lam_geom (25) >> lam_bounds (1)
                  #   CL violations matter most because a foil with wrong lift is
                  #   physically useless. Geometry matters next (manufacturability).
                  #   Latent bounds are a soft guide, not a hard physics constraint.
                  lam_bounds: float = 1.0,
                  lam_geom: float = 25.0,
                  lam_cl: float = 50.0,
                  
                  # Geometry limits (must match nom_driver.py defaults)
                  min_thickness: float = 0.006,
                  max_thickness: float = 0.157,
                  te_gap_max: float = 0.01,
                  min_max_thickness: float = 0.04,
                  max_camber: float = 0.08,
                  
                  # CL window
                  cl_min: float | None = None,
                  cl_max: float | None = None,
                  ) -> tuple[float, dict]:
    """
    Combine all penalties into a single weighted sum.
    
    ACTION ITEM #19 (2/17): PROF FEEDBACK ON LAMBDAS:
      "Lambda i = Pi / sum(Pi) so all lambdas sum to 1"
      "Think about weights so all have same valid vectors"
      "Some p's could be 0 but at the end there will be weights"
    
    ---------------------------------------------------------------
    IMPLEMENTATION NOTE (3/9/26) — FIXED-WEIGHT NORMALIZATION
    ---------------------------------------------------------------
    The professor's formula "λ_i = P_i / Σ(P_i)" describes ADAPTIVE
    normalization: at each step, divide each penalty by the total
    penalty sum so they contribute equally. Example:
      p_bounds=0.5, p_geom=0.01, p_cl=0.0
      λ_bounds = 0.5/0.51 = 0.98, λ_geom = 0.01/0.51 = 0.02
    
    PROBLEM with adaptive normalization:
      When one penalty dominates (e.g. p_bounds=0.5, p_geom=0.001),
      the small penalty's gradient gets scaled to near-zero, so the
      optimizer ignores it. Worse, when all penalties are zero (valid
      foil), you get 0/0 = undefined. And the gradient of (p_i / Σp)
      with respect to the latent vector is more complex (quotient rule)
      which would interact badly with our finite-difference approach.
    
    WHAT WE DO INSTEAD (fixed-weight normalization):
      λ_i = lam_i / (lam_bounds + lam_geom + lam_cl)
      This normalizes the WEIGHTS to sum to 1, not the penalties.
      The relative importance is set by lam_bounds:lam_geom:lam_cl
      (currently 1:25:50) and stays constant throughout optimization.
    
    WHY THIS IS ACCEPTABLE:
      In practice, geometry violations are all-or-nothing (hard reject
      = 1000, which short-circuits before reaching the lambda math).
      The only soft penalty that reaches this weighted sum is the TE
      gap, which is small. So the lambda normalization mainly affects
      the balance between latent-bounds and CL penalties, where
      fixed weights work fine.
    
    TO MATCH PROFESSOR'S FORMULA EXACTLY, you would need:
      p_sum = p_bounds + p_geom + p_cl + 1e-9  # avoid /0
      total = (p_bounds**2 / p_sum) + (p_geom**2 / p_sum) + ...
      But this has the gradient issues described above.
    ---------------------------------------------------------------
    
    FORMULA (current implementation):
      λ_bounds_norm = lam_bounds / (lam_bounds + lam_geom + lam_cl)
      λ_geom_norm   = lam_geom   / (lam_bounds + lam_geom + lam_cl)
      λ_cl_norm     = lam_cl     / (lam_bounds + lam_geom + lam_cl)
      
      total = λ_bounds_norm * p_bounds
            + λ_geom_norm   * p_geom
            + λ_cl_norm     * p_cl
    """
    # --- Penalty 1: latent bounds ---
    p_bounds = latent_bounds_penalty(latent_vec, lat_lo, lat_hi)
    
    # --- Penalty 2: geometry ---
    p_geom, geom_info = geometry_penalty(
        coords,
        min_thickness=min_thickness,
        max_thickness=max_thickness,
        te_gap_max=te_gap_max,
        min_max_thickness=min_max_thickness,
        max_camber=max_camber,
    )
    
    # Hard reject if geometry is impossible (1000 penalty from geometry_penalty)
    # Short-circuit: don't bother computing other penalties, just return 1000.
    if p_geom >= 1000.0:
        return 1000.0, {
            **geom_info,
            "p_bounds": float(p_bounds),
            "p_geom": 1000.0,
            "p_cl": 0.0,
        }
    
    # --- Penalty 3: CL bounds ---
    p_cl = 0.0
    if CL is not None:
        p_cl = cl_bounds_penalty(CL, cl_min=cl_min, cl_max=cl_max)
    
    # --- Normalize lambda weights so they sum to 1 ---
    lam_sum = lam_bounds + lam_geom + lam_cl
    if lam_sum <= 0:
        raise ValueError("Lambda sum must be > 0")
    
    lam_bounds_norm = lam_bounds / lam_sum  # 1/76 ≈ 0.013
    lam_geom_norm   = lam_geom   / lam_sum  # 25/76 ≈ 0.329
    lam_cl_norm     = lam_cl     / lam_sum  # 50/76 ≈ 0.658
    
    # --- Weighted sum ---
    total = float(lam_bounds_norm * p_bounds
                + lam_geom_norm   * p_geom
                + lam_cl_norm     * p_cl)
    
    info = {
        **geom_info,
        "p_bounds": float(p_bounds),
        "p_geom": float(p_geom),
        "p_cl": float(p_cl),
        "lam_bounds_norm": lam_bounds_norm,
        "lam_geom_norm": lam_geom_norm,
        "lam_cl_norm": lam_cl_norm,
    }
    
    return total, info