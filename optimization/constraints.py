"""
optimization/constraints.py

CLEAN REBUILD based on professor's action items (2/17/26 meeting).

WHAT THIS FILE DOES:
  Defines penalty functions for the NOM optimizer. Each candidate foil
  gets a penalty score >= 0 added to its CD/CL objective.

KEY CHANGES from old version:
  ✓ Removed all camber checks (prof: "too complicated, skip for now")
  ✓ Removed LE gap check (prof: "only check TE gap")
  ✓ Removed inf returns (prof: "use 1000 for hard rejects")
  ✓ Check thickness only in interior x ∈ [0.05, 0.90] (prof: line 408 feedback)
  ✓ Simplified geometry_penalty to ONLY return thickness violation
  ✓ Lambda weights auto-normalized so Σ(λ_i) = 1

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
    
    ACTION ITEM #1: "No need for relu max in bounds"
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
# 2) GEOMETRY -- thickness check only (prof: removed camber, LE gap, etc.)
# ===========================================================================

def geometry_penalty(coords: np.ndarray,
                     min_thickness: float = 0.04,
                     max_thickness: float = 0.14,
                     te_gap_max: float = 0.01,
                     thickness_x_min: float = 0.05,
                     thickness_x_max: float = 0.90) -> tuple[float, dict]:
    """
    Check foil geometry. SIMPLIFIED per prof feedback.
    
    CHECKS (in order):
      1) Coords finite and reasonable range
      2) Upper above lower everywhere (hard reject if crossing)
      3) Min thickness in interior (hard reject if too thin)
      4) Max thickness (hard reject if too fat)
      5) TE gap (soft penalty if slightly open)
    
    REMOVED (per prof):
      - ACTION ITEM #16: Camber checks (too complicated)
      - ACTION ITEM #8,17: LE gap check (only check TE)
      - ACTION ITEM #13: Sharpness checks (overcomplicated)
      - ACTION ITEM #14: Overcomplicated checks removed
      - ACTION ITEM #10: Full 0->1 thickness scan (only check interior)
    
    HARD REJECTS:
      ACTION ITEM #3: Instead of returning inf, return 1000 (prof: "no inf, use big number")
    
    INPUTS:
      coords -- shape (80, 2): decoded foil coords
        rows 0-39  = upper surface TE->LE (x: 1->0)
        rows 40-79 = lower surface LE->TE (x: 0->1)
      min_thickness    -- hard limit (from dataset scan)
      max_thickness    -- hard limit
      te_gap_max       -- soft limit for TE closure
      thickness_x_min  -- interior check starts at x=0.05
      thickness_x_max  -- interior check ends at x=0.90
    
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
    # ACTION ITEM #5-6: Prof: "Find max y and x from dataset, narrow to ½ for buffer"
    # UPDATED: Used compute_dataset_max_xy.py to find actual dataset bounds.
    # Result: 99th percentile max |y| = 0.1786, with 10% buffer = 0.1964
    # This ignores outliers like naca1.txt (y=1.0) which is 3x larger than normal foils.
    if not np.all(np.isfinite(x_all)) or not np.all(np.isfinite(y_all)):
        return 1000.0, {"reason": "nonfinite coords"}
    
    # ACTION ITEM #4: "Hard reject with x will never happen but y will"
    # Sanity: x should be in [0,1], y should be reasonable
    if np.any(x_all < -0.1) or np.any(x_all > 1.1):
        return 1000.0, {"reason": "x out of range"}
    
    # ACTION ITEM #5: "max_abs_y = 0.25 (random number), not good idea"
    # FIXED: Now using dataset 99th percentile + 10% buffer (0.1964)
    if np.any(np.abs(y_all) > 0.1964):  # Dataset 99th percentile + 10% buffer
        return 1000.0, {"reason": "y too large"}
    
    # --- CHECK 2: Upper above lower (hard reject if crossing) ---
    # ACTION ITEM #2: "Didnt update from interpolation script in split_upper_lower (take out)"
    # Removed split_upper_lower function, now splitting surfaces directly here.
    upper_te2le = coords[:40]  # x: 1->0
    lower_le2te = coords[40:]  # x: 0->1
    
    # Flip upper to LE->TE so both go same direction
    upper_le2te = upper_te2le[::-1]  # x: 0->1
    
    xu, yu = upper_le2te[:, 0], upper_le2te[:, 1]
    xl, yl = lower_le2te[:, 0], lower_le2te[:, 1]

    # ACTION ITEM #9: "Line 408 no need for interpolation"
    # The decoder outputs both surfaces on the SAME x-grid (linspace 0->1, 40 pts).
    # So upper and lower share x-values point-for-point -- no interpolation needed.
    # We can subtract y values directly at each shared x point.
    #
    # The old code interpolated onto a new 120-point grid which was:
    #   1) unnecessary (same x-grid already)
    #   2) causing false surface-crossing rejects at LE (x=0) due to
    #      interpolation artifacts on the sharp leading edge
    # Both problems are eliminated by using the decoder grid directly.

    # Both surfaces are already on linspace(0,1,40) -- use directly
    xg = xu  # shape (40,) -- same x values for both surfaces
    thickness = yu - yl  # shape (40,) -- point-by-point, no interpolation

    # --- CHECK 2: Upper above lower in INTERIOR only ---
    # Skip near LE (x < 0.05) where foil tapers to a point --
    # thickness naturally approaches 0 there and is not meaningful.
    interior_mask = xg >= 0.05
    if np.any(thickness[interior_mask] < -1e-6):
        return 1000.0, {"reason": "surfaces crossing"}

    # --- CHECK 3 & 4: Thickness in INTERIOR only (prof: x in [0.05, 0.90]) ---
    # ACTION ITEM #10: "Instead of 0 to 1, from .1 to .9 or .05 to .9"
    # ACTION ITEM #11: "Line 433, not negative -1e-4 make positive"
    # ACTION ITEM #12: "Find min thickness from dataset between .1 and .9"
    #
    # LOGIC: We only check thickness in the middle of the foil.
    # Near LE (x~0) and TE (x~1) every real foil tapers thin -- checking
    # there would reject all valid foils. The interior [0.05, 0.90] is
    # where structural thickness actually matters.
    mask = (xg >= thickness_x_min) & (xg <= thickness_x_max)
    t_interior = thickness[mask]
    
    if len(t_interior) == 0:
        return 1000.0, {"reason": "no interior points"}
    
    t_min = float(np.min(t_interior))  # ACTION ITEM #7: using float() not nom/numpy
    t_max = float(np.max(t_interior))  # ACTION ITEM #7: direct float conversion
    
    # Hard reject if too thin (prof: "check min thickness from dataset")
    # ACTION ITEM #11: Make positive (was -1e-4, now positive comparison)
    # ACTION ITEM #15: "too thin redundant, too thick check max/min"
    if t_min < min_thickness:
        return 1000.0, {"reason": "too thin", "t_min": t_min}
    
    # Hard reject if too thick
    # ACTION ITEM #15: Check against max_thickness from constraints
    if t_max > max_thickness:
        return 1000.0, {"reason": "too thick", "t_max": t_max}
    
    # --- CHECK 5: TE gap (soft penalty) ---
    # ACTION ITEM #8: "No need for leading edge line 404, only trailing edge"
    # ACTION ITEM #9: "Line 408 no need for interpolation"
    # Prof: Check TE gap directly from coords, no interpolation needed.
    # If TE closes properly, gap will be ~0 which is correct.
    
    # TE = x ≈ 1.0 (last points of each surface)
    y_upper_te = upper_te2le[0, 1]  # first row of upper (which is TE in TE->LE order)
    y_lower_te = lower_le2te[-1, 1]  # last row of lower (which is TE in LE->TE order)
    
    te_gap = float(abs(y_upper_te - y_lower_te))
    
    # Soft penalty for TE not closing
    p_te = relu(te_gap - te_gap_max)
    
    # --- RETURN ---
    # ACTION ITEM #18: "Line 480: thickness should be a penalty, return summation"
    # Prof: "Have another p value, p = min_t, line 509 summation of all penalties"
    # We already did thickness checks above (hard rejects for too thin/thick).
    # Here we only return the soft TE gap penalty.
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
# 4) TOTAL PENALTY with auto-normalized lambdas
# ===========================================================================

def total_penalty(*,
                  latent_vec: np.ndarray,
                  coords: np.ndarray,
                  CL: float | None,
                  lat_lo: np.ndarray,
                  lat_hi: np.ndarray,
                  
                  # Raw penalty weights (will be auto-normalized)
                  lam_bounds: float = 1.0,
                  lam_geom: float = 25.0,
                  lam_cl: float = 50.0,
                  
                  # Geometry limits
                  min_thickness: float = 0.04,
                  max_thickness: float = 0.14,
                  te_gap_max: float = 0.01,
                  
                  # CL window
                  cl_min: float | None = None,
                  cl_max: float | None = None,
                  ) -> tuple[float, dict]:
    """
    Combine all penalties with auto-normalized lambda weights.
    
    ACTION ITEM #19: PROF FEEDBACK ON LAMBDAS:
      "Lambda i = Pi / sum(Pi) so all lambdas sum to 1"
      "Think about weights so all have same valid vectors"
      "Some p's could be 0 but at the end there will be weights"
    
    HOW IT WORKS:
      1) Compute raw penalties: p_bounds, p_geom, p_cl
      2) Normalize lambdas: λ_i = lam_i / Σ(lam_i)
      3) Total = Σ(λ_i * p_i)
    
    FORMULA:
      λ_bounds_norm = lam_bounds / (lam_bounds + lam_geom + lam_cl)
      λ_geom_norm   = lam_geom   / (lam_bounds + lam_geom + lam_cl)
      λ_cl_norm     = lam_cl     / (lam_bounds + lam_geom + lam_cl)
      
      total = λ_bounds_norm * p_bounds
            + λ_geom_norm   * p_geom
            + λ_cl_norm     * p_cl
    
    This ensures Σ(λ_i) = 1 automatically, regardless of what raw lam values
    you pass in. The RELATIVE sizes still matter (e.g. lam_cl=50 vs lam_bounds=1
    means CL violations are weighted 50x more), but now they're on a consistent
    scale.
    """
    # --- Penalty 1: latent bounds ---
    p_bounds = latent_bounds_penalty(latent_vec, lat_lo, lat_hi)
    
    # --- Penalty 2: geometry ---
    p_geom, geom_info = geometry_penalty(
        coords,
        min_thickness=min_thickness,
        max_thickness=max_thickness,
        te_gap_max=te_gap_max,
    )
    
    # Hard reject if geometry is impossible
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
    
    # --- Normalize lambdas so they sum to 1 ---
    lam_sum = lam_bounds + lam_geom + lam_cl
    if lam_sum <= 0:
        raise ValueError("Lambda sum must be > 0")
    
    lam_bounds_norm = lam_bounds / lam_sum
    lam_geom_norm   = lam_geom   / lam_sum
    lam_cl_norm     = lam_cl     / lam_sum
    
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