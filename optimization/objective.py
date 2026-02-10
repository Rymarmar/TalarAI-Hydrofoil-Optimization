import numpy as np

# Purpose:
#   Define "what we are trying to optimize" (performance only)
#   Constraints are NOT handled here. Constraints live in constraints.py
#
# Key idea:
#   We want good L/D. Minimizing CD/CL is equivalent to maximizing CL/CD
#   We avoid abs(CL) now because we explicitly constrain CL with penalties


def cd_over_abscl(CL: float, CD: float, eps: float = 1e-9) -> float:
    """
    Sponsor-safe / sign-invariant objective.
    Minimizes CD / |CL| so negative CL doesn't crash the objective.
    We keep this around for debugging/comparison.
    """
    return float(CD) / max(float(abs(CL)), eps)


def cd_over_cl(CL: float, CD: float, eps: float = 1e-9) -> float:
    """
    Physical objective:
      minimize CD/CL (maximize CL/CD).
    If CL <= 0, we return a huge value because it's not physically acceptable
    for the chosen coordinate convention (positive lift only).
    """
    CL = float(CL)
    CD = float(CD)

    # basic safety checks
    if not np.isfinite(CL) or not np.isfinite(CD):
        return float("inf")

    # if CL is negative/near zero, this is "bad", so make objective huge
    if CL <= eps:
        return 1e9

    return CD / max(CL, eps)


def default_objective(CL: float, CD: float) -> float:
    """
    This is the objective used by NOM.
    We use CD/CL because constraints.py now enforces a valid CL range.
    """
    return cd_over_cl(CL, CD)
