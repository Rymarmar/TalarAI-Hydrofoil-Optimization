"""
objective.py

Performance objective ONLY (no constraints here).

What we optimize:
  minimize  CD / CL
This is equivalent to maximizing CL / CD, but writing it as CD/CL keeps the
"minimize" direction consistent with our NOM loop.

Why no huge penalties here?
- Constraints belong in constraints.py (NOM style: objective + lambdas*ReLU(violations))
- If CL <= 0, CD/CL stops making physical sense for "positive lift" designs,
  so we return +inf and let constraints/optimizer reject it naturally.
"""

from __future__ import annotations

import math


def cd_over_cl(CL: float, CD: float, eps: float = 1e-9) -> float:
    CL = float(CL)
    CD = float(CD)

    # Keep this file lightweight: we only need finiteness checks.
    if not (math.isfinite(CL) and math.isfinite(CD)):
        return float("inf")

    # Meeting note (Objective action item):
    # The core objective should stay as CD/(CL+eps). No other hidden terms.
    # We only require CL > 0 so the ratio is physically meaningful.
    if CL <= 0.0:
        return float("inf")

    return CD / (CL + eps)


def default_objective(CL: float, CD: float) -> float:
    return cd_over_cl(CL, CD)
