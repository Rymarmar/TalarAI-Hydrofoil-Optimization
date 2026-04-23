"""
optimization/nom_driver.py

CLEAN REBUILD based on professor's action items (2/17/26 meeting).

WHAT THIS FILE DOES:
  Defines penalty functions for the NOM optimizer. Each candidate foil
  gets a penalty score >= 0 added to its CD/CL objective.

KEY CHANGES from old version:
  ✓ Camber check added back (2/19: "no cambered foils, hard to 3D print")
  ✓ Removed LE gap check (prof: "only check TE gap")
  ✓ Removed inf returns (prof: "use 1000 for hard rejects")
  ✓ Lambda weights auto-normalized so Σ(λ_i) = 1

FIXES APPLIED (3/9/26):
  ✓ FIX 1: Surface-crossing check now uses [0.05, 0.90] interior mask,
    matching the thickness check range.

  ✓ FIX 2: Lambda normalization documented as FIXED-WEIGHT normalization,
    not the adaptive normalization the professor described.

THICKNESS OVERHAUL (4/6/26 meeting action items):
  ✓ Replaced single min_thickness with THREE ZONE checks:
      - LE  zone: x ∈ [0.05, 0.15]  min = min_thickness_le
      - Mid zone: x ∈ [0.15, 0.75]  min = min_thickness_mid
      - TE  zone: x ∈ [0.75, 0.95]  min = min_thickness_te
    The TE zone now extends to x=0.95 (was 0.90) to catch the razor-edge
    region the optimizer was exploiting.

  ✓ Added MINIMUM TE WEDGE ANGLE check (hard reject):
    Computes the full wedge angle at the trailing edge from the last 3
    grid points of each surface. If angle < min_te_angle_deg → reject.
    This directly prevents the sharp TE seen in the 4/6/26 output plot.
    NACA 0012 has a TE wedge angle of ~15.5 deg; floor is set at 14 deg
    per professor's physical constraint (4/13/26).

  ✓ x_peak_t (x-location of max thickness) now returned in info dict.
    Useful for reporting where the metal rods would go in the physical rig.

DEFAULT THRESHOLDS (calibrated to NACA 0012 with ~30-40% headroom):
  NACA 0012 actual thickness at zone boundaries:
    x=0.05:  7.1%c   → LE floor: 2.5%c (0.025)
    x=0.75:  6.3%c   → Mid floor: 3.0%c (0.030)
    x=0.95:  1.6%c   → TE floor: 0.8%c (0.008)
    TE angle: 15.5°   → angle floor: 14° (prof 4/13/26 physical constraint)
  Change these numbers once physical manufacturing constraints are known.

PENALTIES:
  1) latent_bounds_penalty  -- params outside training data range
  2) geometry_penalty       -- thickness / angle / camber violations
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
    z  = np.asarray(latent_vec, dtype=float).reshape(-1)
    lo = np.asarray(lo,         dtype=float).reshape(-1)
    hi = np.asarray(hi,         dtype=float).reshape(-1)

    if z.shape[0] != 6 or lo.shape[0] != 6 or hi.shape[0] != 6:
        raise ValueError("All arrays must be length 6")

    pen = 0.0
    for i in range(6):
        pen += relu(lo[i] - z[i])   # went below min
        pen += relu(z[i] - hi[i])   # went above max

    return float(pen)


# ===========================================================================
# 2) GEOMETRY -- thickness, TE angle, camber, TE gap checks
# ===========================================================================

class _ThicknessHardReject(Exception):
    """Raised internally when a thickness violation is too extreme to penalize softly."""
    def __init__(self, zone, t_min, floor):
        self.zone, self.t_min, self.floor = zone, t_min, floor


def geometry_penalty(coords: np.ndarray,
                     # ----------------------------------------------------------
                     # THREE-ZONE MINIMUM THICKNESS  (action item 4/6/26)
                     #
                     # WHY THREE ZONES instead of one:
                     #   A single min_thickness applied to [0.05, 0.90] left
                     #   the last 10% of chord (x=0.90→1.0) completely
                     #   unchecked. The optimizer legally drove the TE to a
                     #   near-razor edge. Splitting into three zones lets us
                     #   apply the right floor for each region.
                     #
                     # ZONE DEFINITIONS:
                     #   LE  zone: x ∈ [0.05, 0.15]  — foil is naturally thin here
                     #   Mid zone: x ∈ [0.15, 0.75]  — structural core
                     #   TE  zone: x ∈ [0.75, 0.95]  — KEY: extended from old 0.90
                     #                                  to 0.95 to catch razor TE
                     #
                     # DEFAULT VALUES (calibrated to NACA 0012, ~30-40% headroom):
                     #   NACA 0012 actual at x=0.05: 7.1%c  → floor: 2.5%c
                     #   NACA 0012 actual at x=0.75: 6.3%c  → floor: 3.0%c
                     #   NACA 0012 actual at x=0.95: 1.6%c  → floor: 0.8%c
                     #
                     # TO ADJUST FOR MANUFACTURING:
                     #   Once you know the minimum 3D-printable wall thickness
                     #   (in mm) and the physical chord length, convert to %c
                     #   and plug in here. The zone structure stays the same,
                     #   you only change the numbers.
                     # ----------------------------------------------------------
                     min_thickness_le:  float = 0.025,  # LE  zone: x ∈ [0.05, 0.15]
                     min_thickness_mid: float = 0.030,  # Mid zone: x ∈ [0.15, 0.75]
                     min_thickness_te:  float = 0.008,  # TE  zone: x ∈ [0.75, 0.95]
                     # ----------------------------------------------------------
                     # MAXIMUM THICKNESS (global cap, same as before)
                     # ----------------------------------------------------------
                     max_thickness: float = 0.157,
                     # ----------------------------------------------------------
                     # TE GAP (soft penalty for open trailing edge)
                     # ----------------------------------------------------------
                     te_gap_max: float = 0.01,
                     # ----------------------------------------------------------
                     # MIN-MAX THICKNESS (peak must be structurally deep enough)
                     # ACTION ITEM (2/19): hard limit on the peak value
                     # ----------------------------------------------------------
                     min_max_thickness: float = 0.04,
                     # ----------------------------------------------------------
                     # MAX CAMBER
                     # REVISION (4/1/26): tightened from 0.10 → 0.06 (6%c).
                     # ----------------------------------------------------------
                     max_camber: float = 0.06,
                     # ----------------------------------------------------------
                     # LE Y-POSITION CHECK (action item 4/1/26)
                     # ----------------------------------------------------------
                     max_le_y: float = 0.01,
                     # ----------------------------------------------------------
                     # MINIMUM TE WEDGE ANGLE  (action item 4/6/26)
                     #
                     # WHY THIS IS NEEDED:
                     #   The TE zone thickness floor prevents very thin walls,
                     #   but a foil can still taper to a sharp point at x=1.0
                     #   between x=0.95 and x=1.0. The wedge angle check catches
                     #   this directly: if the two surfaces are converging too
                     #   steeply, the foil will be physically fragile and hard
                     #   to manufacture.
                     #
                     # HOW IT IS COMPUTED:
                     #   Use the last 3 points of each surface (near x=1.0) to
                     #   estimate the slope of each surface approaching the TE.
                     #   Full wedge angle = arctan(|upper_slope|) + arctan(|lower_slope|)
                     #   For NACA 0012 on a 40-pt grid: ~15.5 degrees.
                     #
                     # REVISION (4/13/26): raised from 6 → 14 deg based on
                     #   professor's physical constraint (0.04in wall at c/40 from TE).
                     # ----------------------------------------------------------
                     min_te_angle_deg: float = 14.0,
                     # ----------------------------------------------------------
                     # THICKNESS SOFT PENALTY WEIGHT
                     # Controls how hard the gradient pushes toward thicker foils
                     # when below the zone floor. Higher = stronger push.
                     # Default 200: a 1%c violation adds 200*0.01 = 2.0 to loss,
                     # comparable to the CD/CL objective (~0.03).
                     # ----------------------------------------------------------
                     lam_thickness: float = 200.0,
                     # Maximum absolute y value — dataset 99th percentile is 0.1964.
                     # Auto-calibrated to max(1.05 * baseline_max_abs_y, 0.1964)
                     # so drawn foils that decode to thicker shapes still pass.
                     max_y_abs: float = 0.1964,
                     ) -> tuple[float, dict]:
    """
    Check foil geometry and return (penalty, info_dict).

    CHECKS (in order):
      1) Coords finite and reasonable range
      2) Upper above lower in interior [0.05, 0.95] (hard reject if crossing)
      3) THREE-ZONE min thickness: LE [0.05,0.15], Mid [0.15,0.75], TE [0.75,0.95]
      4) Max thickness (hard reject if too fat)
      5) Min-max thickness (hard reject if peak thickness < min_max_thickness)
      6) Max camber (hard reject if too cambered)
      7) TE wedge angle (hard reject if TE too sharp)
      8) LE y-position (hard reject if nose lifted off chord line)
      9) TE gap (soft penalty if slightly open)

    HARD REJECTS return penalty = 1000.
    SOFT VIOLATIONS return a small float.
    """
    coords = np.asarray(coords, dtype=float)

    if coords.shape != (80, 2):
        return 1000.0, {"reason": "coords wrong shape"}

    x_all, y_all = coords[:, 0], coords[:, 1]

    # --- CHECK 1: Coords finite and in reasonable range ---
    if not np.all(np.isfinite(x_all)) or not np.all(np.isfinite(y_all)):
        return 1000.0, {"reason": "nonfinite coords"}

    if np.any(x_all < -0.1) or np.any(x_all > 1.1):
        return 1000.0, {"reason": "x out of range"}

    # Dataset 99th percentile max |y| + 10% buffer
    if np.any(np.abs(y_all) > max_y_abs):
        return 1000.0, {"reason": "y too large"}

    # --- SURFACE SPLITTING ---
    upper_te2le = coords[:40]         # x: 1->0 (TE to LE)
    lower_le2te = coords[40:]         # x: 0->1 (LE to TE)
    upper_le2te = upper_te2le[::-1]   # flip to x: 0->1 for comparison

    xu, yu = upper_le2te[:, 0], upper_le2te[:, 1]
    xl, yl = lower_le2te[:, 0], lower_le2te[:, 1]

    # Both surfaces are on linspace(0,1,40) -- direct subtraction, no interpolation
    xg        = xu                    # shape (40,)
    thickness = yu - yl               # shape (40,) -- point-by-point

    # --- CHECK 2: Surface crossing in extended interior [0.05, 0.95] ---
    # Extended to 0.95 (was 0.90) to match the new TE zone boundary.
    crossing_mask = (xg >= 0.05) & (xg <= 0.95)
    if np.any(thickness[crossing_mask] < -1e-6):
        return 1000.0, {"reason": "surfaces crossing"}

    # ===========================================================================
    # CHECKS 3 & 4: THREE-ZONE MINIMUM THICKNESS  (action item 4/6/26)
    # ===========================================================================
    # Each zone is checked independently. The first zone to fail hard-rejects.
    # LE and Mid zones use the same x-boundaries as the crossing check.
    # TE zone extends to x=0.95 -- this is the key change that prevents razor TE.

    # LE zone: x ∈ [0.05, 0.15]
    le_mask = (xg >= 0.05) & (xg <= 0.15)
    t_le    = thickness[le_mask]
    if len(t_le) > 0:
        t_le_min = float(np.min(t_le))
    else:
        t_le_min = float("nan")

    # Mid zone: x ∈ (0.15, 0.75]
    mid_mask = (xg > 0.15) & (xg <= 0.75)
    t_mid    = thickness[mid_mask]
    if len(t_mid) > 0:
        t_mid_min = float(np.min(t_mid))
    else:
        t_mid_min = float("nan")

    # TE zone: x ∈ (0.75, 0.95]
    te_zone_mask = (xg > 0.75) & (xg <= 0.95)
    t_te         = thickness[te_zone_mask]
    if len(t_te) > 0:
        t_te_min = float(np.min(t_te))
    else:
        t_te_min = float("nan")

    # Zone thickness soft penalties
    # Each violation = how much thinner the foil is than the floor, scaled by
    # lam_thickness so the gradient actively pushes toward thicker foils.
    # Hard-reject (1000) only fires when the violation is extreme (>50% of floor).
    p_thickness = 0.0
    thickness_reason = "ok"

    def _zone_pen(t_min_val, floor, zone_name):
        nonlocal p_thickness, thickness_reason
        if np.isfinite(t_min_val) and t_min_val < floor:
            violation = floor - t_min_val
            if violation > 0.5 * floor:
                # Extreme violation — foil is less than half the required thickness.
                # Hard-reject so we don't waste FD budget on impossible shapes.
                raise _ThicknessHardReject(zone_name, t_min_val, floor)
            p_thickness += lam_thickness * violation
            if thickness_reason == "ok":
                thickness_reason = f"too thin ({zone_name})"

    try:
        _zone_pen(t_le_min,  min_thickness_le,  "LE zone")
        _zone_pen(t_mid_min, min_thickness_mid, "Mid zone")
        _zone_pen(t_te_min,  min_thickness_te,  "TE zone")
    except _ThicknessHardReject as e:
        return 1000.0, {
            "reason": f"too thin ({e.zone}) — extreme violation",
            "t_min":  e.t_min,
            "floor":  e.floor,
            "zone":   e.zone,
        }

    # Summary min/max over the full interior for reporting and the checks below
    interior_mask = (xg >= 0.05) & (xg <= 0.95)
    t_interior    = thickness[interior_mask]

    if len(t_interior) == 0:
        return 1000.0, {"reason": "no interior points"}

    t_min = float(np.min(t_interior))
    t_max = float(np.max(t_interior))

    # --- CHECK 4 (old CHECK 4): Max thickness cap ---
    if t_max > max_thickness:
        return 1000.0, {"reason": "too thick", "t_max": t_max}

    # --- CHECK 5: Min-max thickness (peak must be structurally deep) ---
    if t_max < min_max_thickness:
        return 1000.0, {
            "reason":            "peak too thin",
            "t_max":             t_max,
            "min_max_thickness": min_max_thickness,
        }

    # x-location of max thickness (returned in info dict for reporting)
    t_peak_idx = int(np.argmax(t_interior))
    x_peak_t   = float(xg[interior_mask][t_peak_idx])

    # --- CHECK 6: Max camber ---
    camber_line     = (yu + yl) / 2.0
    camber_interior = camber_line[interior_mask]
    max_camber_actual = float(np.max(np.abs(camber_interior)))
    if max_camber_actual > max_camber:
        return 1000.0, {
            "reason":           "too cambered",
            "max_camber_actual": max_camber_actual,
            "max_camber_limit":  max_camber,
        }

    # ===========================================================================
    # CHECK 7: MINIMUM TE WEDGE ANGLE  (action item 4/6/26)
    # ===========================================================================
    # Compute the slope of each surface using the last 3 grid points (x ≈ 0.95→1.0).
    # Upper surface approaches TE with a negative slope (going down).
    # Lower surface approaches TE with a positive slope (going up).
    # Full wedge angle = sum of both absolute arctangents.
    #
    # NACA 0012 (40-pt grid) gives ~15.5 deg.
    # Floor raised to 14 deg per professor's physical constraint (4/13/26).

    dx_te = float(xu[-1] - xu[-3])   # x spacing over last 3 pts (~0.051)
    if dx_te > 1e-9:
        upper_te_slope = (yu[-1] - yu[-3]) / dx_te   # negative for normal upper
        lower_te_slope = (yl[-1] - yl[-3]) / dx_te   # positive for normal lower
        te_angle_deg   = float(np.degrees(
            np.arctan(abs(upper_te_slope)) + np.arctan(abs(lower_te_slope))
        ))
    else:
        te_angle_deg = 0.0  # degenerate grid -- treat as zero angle

    if te_angle_deg < min_te_angle_deg:
        return 1000.0, {
            "reason":          "TE too sharp",
            "te_angle_deg":    te_angle_deg,
            "min_te_angle_deg": min_te_angle_deg,
        }

    # --- CHECK 8: LE must stay at y ≈ 0 (action item 4/1/26) ---
    le_y_upper = float(yu[0])
    le_y_lower = float(yl[0])
    le_y_avg   = (le_y_upper + le_y_lower) / 2.0

    if abs(le_y_avg) > max_le_y:
        return 1000.0, {
            "reason":     "LE too far from y=0",
            "le_y_avg":   le_y_avg,
            "le_y_upper": le_y_upper,
            "le_y_lower": le_y_lower,
            "max_le_y":   max_le_y,
        }

    # --- CHECK 9: TE gap (soft penalty) ---
    y_upper_te = upper_te2le[0, 1]    # first row of upper (TE in TE->LE order)
    y_lower_te = lower_le2te[-1, 1]   # last row of lower (TE in LE->TE order)
    te_gap     = float(abs(y_upper_te - y_lower_te))
    p_te       = relu(te_gap - te_gap_max)

    # --- RETURN ---
    total_pen = float(p_te + p_thickness)

    info = {
        # Zone mins for visibility in logs/plots
        "t_le_min":     t_le_min,
        "t_mid_min":    t_mid_min,
        "t_te_min":     t_te_min,
        # Overall interior summary (used by old callers that read t_min/t_max)
        "t_min":        t_min,
        "t_max":        t_max,
        # Max thickness location (useful for physical rig / metal rod placement)
        "x_peak_t":     x_peak_t,
        # TE diagnostics
        "te_angle_deg": te_angle_deg,
        "te_gap":       te_gap,
        "p_te":         float(p_te),
        "p_thickness":  float(p_thickness),
        "reason":       "ok" if total_pen == 0 else thickness_reason if p_thickness > 0 else "soft_violations",
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
    """
    CL = float(CL)

    if not np.isfinite(CL):
        return 1000.0

    pen = 0.0
    if cl_min is not None:
        pen += relu(float(cl_min) - CL)
    if cl_max is not None:
        pen += relu(CL - float(cl_max))

    return float(pen)


# ===========================================================================
# 3b) ROD PENALTY  -- dedicated quadratic penalty for structural rod constraints
# ===========================================================================

def rod_penalty(coords: np.ndarray,
                rods: list,
                lam_rod: float = 500.0) -> tuple:
    """
    Quadratic penalty for foil being thinner than each rod diameter at its x station.

    For each rod: if foil_thickness_at_x < rod_diam, adds lam_rod*(violation/rod_diam)^2.
    Quadratic growth gives a strong gradient signal pushing toward thicker foils,
    unlike linear which the L/D objective can easily overpower.
    Checked at the exact rod x-station, not zone-wide, for a precise gradient.
    """
    if not rods:
        return 0.0, {"p_rod": 0.0, "rod_violations": []}

    coords = np.asarray(coords, dtype=float)
    if coords.shape != (80, 2):
        return 0.0, {"p_rod": 0.0, "rod_violations": []}

    upper_le2te = coords[:40][::-1]
    lower_le2te = coords[40:]
    xg        = upper_le2te[:, 0]
    thickness = upper_le2te[:, 1] - lower_le2te[:, 1]

    total_pen  = 0.0
    violations = []

    for rod in rods:
        rod_x    = float(rod["x"])
        rod_diam = float(rod["diam"])
        if rod_diam <= 0:
            continue
        idx     = int(np.argmin(np.abs(xg - rod_x)))
        local_t = float(thickness[idx])
        if local_t < rod_diam:
            violation      = rod_diam - local_t
            frac_violation = violation / rod_diam
            pen            = lam_rod * frac_violation ** 2
            total_pen     += pen
            violations.append({"x": rod_x, "rod_diam": rod_diam,
                                "foil_t": local_t, "violation": violation, "pen": pen})
        else:
            violations.append({"x": rod_x, "rod_diam": rod_diam,
                                "foil_t": local_t, "violation": 0.0, "pen": 0.0})

    return float(total_pen), {"p_rod": float(total_pen), "rod_violations": violations}


# ===========================================================================
# 4) TOTAL PENALTY with lambda weights
# ===========================================================================

def total_penalty(*,
                  latent_vec: np.ndarray,
                  coords: np.ndarray,
                  CL: float | None,
                  lat_lo: np.ndarray,
                  lat_hi: np.ndarray,

                  # Penalty weights
                  lam_bounds: float = 1.0,
                  lam_geom:   float = 25.0,
                  lam_cl:     float = 50.0,

                  # Three-zone thickness limits
                  min_thickness_le:  float = 0.025,
                  min_thickness_mid: float = 0.030,
                  min_thickness_te:  float = 0.008,

                  # Other geometry limits
                  max_thickness:     float = 0.157,
                  te_gap_max:        float = 0.01,
                  min_max_thickness: float = 0.04,
                  max_camber:        float = 0.06,
                  max_le_y:          float = 0.02,

                  # TE angle (raised to 14° per prof 4/13/26 physical constraint)
                  min_te_angle_deg:  float = 14.0,

                  # Thickness soft penalty weight (passed to geometry_penalty)
                  lam_thickness: float = 200.0,

                  # Max absolute y — auto-calibrated from baseline for drawn foils
                  max_y_abs: float = 0.1964,

                  # Rod constraints: list of {'x': float, 'diam': float}
                  rods: list | None = None,
                  lam_rod: float = 500.0,

                  # CL window
                  cl_min: float | None = None,
                  cl_max: float | None = None,
                  ) -> tuple[float, dict]:
    """
    Combine all penalties into a single weighted sum.

    FORMULA (fixed-weight normalization):
      λ_i_norm = lam_i / (lam_bounds + lam_geom + lam_cl)
      total = λ_bounds_norm * p_bounds
            + λ_geom_norm   * p_geom
            + λ_cl_norm     * p_cl
    """
    # --- Penalty 1: latent bounds ---
    p_bounds = latent_bounds_penalty(latent_vec, lat_lo, lat_hi)

    # --- Penalty 2: geometry ---
    p_geom, geom_info = geometry_penalty(
        coords,
        min_thickness_le=min_thickness_le,
        min_thickness_mid=min_thickness_mid,
        min_thickness_te=min_thickness_te,
        max_thickness=max_thickness,
        te_gap_max=te_gap_max,
        min_max_thickness=min_max_thickness,
        max_camber=max_camber,
        max_le_y=max_le_y,
        min_te_angle_deg=min_te_angle_deg,
        lam_thickness=lam_thickness,
        max_y_abs=max_y_abs,
    )

    # Hard reject short-circuit
    if p_geom >= 1000.0:
        return 1000.0, {
            **geom_info,
            "p_bounds": float(p_bounds),
            "p_geom":   1000.0,
            "p_cl":     0.0,
        }

    # --- Penalty 3: CL bounds ---
    p_cl = 0.0
    if CL is not None:
        p_cl = cl_bounds_penalty(CL, cl_min=cl_min, cl_max=cl_max)

    # --- Penalty 4: Rod constraints (quadratic, localised at each rod x) ---
    p_rod, rod_info = rod_penalty(coords, rods or [], lam_rod=lam_rod)

    # --- Normalize lambda weights so they sum to 1 ---
    lam_sum = lam_bounds + lam_geom + lam_cl
    if lam_sum <= 0:
        raise ValueError("Lambda sum must be > 0")

    lam_bounds_norm = lam_bounds / lam_sum
    lam_geom_norm   = lam_geom   / lam_sum
    lam_cl_norm     = lam_cl     / lam_sum

    # p_rod is added directly — lam_rod is absolute, not normalised with others
    total = float(lam_bounds_norm * p_bounds
                + lam_geom_norm   * p_geom
                + lam_cl_norm     * p_cl
                + p_rod)

    info = {
        **geom_info,
        **rod_info,
        "p_bounds":        float(p_bounds),
        "p_geom":          float(p_geom),
        "p_cl":            float(p_cl),
        "p_rod":           float(p_rod),
        "lam_bounds_norm": lam_bounds_norm,
        "lam_geom_norm":   lam_geom_norm,
        "lam_cl_norm":     lam_cl_norm,
    }

    return total, info