"""
optimization/objective.py

WHAT THIS FILE DOES:
  Defines the single number we are trying to MINIMIZE: CD / CL.

  CD = drag coefficient   (how much the water resists the foil moving through it)
  CL = lift coefficient   (how much upward force the foil generates)

  Minimizing CD/CL is exactly the same as MAXIMIZING CL/CD (lift-to-drag ratio).
  We write it as CD/CL instead of CL/CD only because our optimizer is built to
  MINIMIZE things (smaller = better), so CD/CL fits naturally.

  WHY L/D (lift-to-drag) matters for a hydrofoil boat:
    A higher L/D means the boat can fly above the water using less engine power.
    For example, L/D = 46 means the foil generates 46x more lift than drag.

WHY CONSTRAINTS ARE NOT HERE:
  Shape constraints (thickness, camber, etc.) live in constraints.py.
  This file ONLY defines the aerodynamic objective. Keeping them separate
  makes it easy to change one without breaking the other.
"""

from __future__ import annotations

import math


def cd_over_cl(CL: float, CD: float, eps: float = 1e-9) -> float:
    """
    Compute CD / CL -- the value we want to MINIMIZE.

    INPUTS:
      CL  -- lift coefficient from NeuralFoil  (dimensionless, typically 0.3 to 1.0)
      CD  -- drag coefficient from NeuralFoil  (dimensionless, typically 0.01 to 0.05)
      eps -- tiny number to prevent divide-by-zero  (1e-9 is negligible)

    OUTPUT:
      CD / (CL + eps) -- a positive float. Smaller = better aerodynamics.
      float("inf")    -- returned when input is bad OR CL <= 0 (no/negative lift).

    WHY eps IN THE DENOMINATOR:
      If CL were exactly 0.0, dividing by zero would crash Python.
      Adding eps = 0.000000001 safely prevents that. It barely changes the
      result: 0.02 / (0.8 + 0.000000001) is still basically 0.02/0.8 = 0.025.

    WHY WE RETURN inf WHEN CL <= 0:
      A foil with zero or negative lift pushes the boat INTO the water
      instead of lifting it. The optimizer should never accept this.
      Returning inf means "skip this foil -- it is physically wrong."
    """
    CL = float(CL)
    CD = float(CD)

    # If NeuralFoil gave us NaN or Infinity, something broke -- skip this foil
    if not (math.isfinite(CL) and math.isfinite(CD)):
        return float("inf")

    # ACTION ITEM (meeting): Comment why we return inf when CL <= 0.
    # We only want foils generating POSITIVE lift (pushing boat UP).
    # Negative CL means the foil is pushing DOWN -- physically wrong.
    if CL <= 0.0:
        return float("inf")

    # ACTION ITEM (meeting): Comment this return line (what the professor asked about).
    # This is the core objective function:  minimize CD / CL
    # This is mathematically identical to maximizing CL / CD (lift-to-drag ratio).
    # Example result: CL=0.84, CD=0.018 --> CD/CL = 0.0214 --> L/D = 1/0.0214 = 46.7
    # The NOM optimizer picks the latent vector that makes this number smallest.
    return CD / (CL + eps)


def default_objective(CL: float, CD: float) -> float:
    """
    This is the only function nom_driver.py calls.
    It wraps cd_over_cl so nom_driver doesn't need to know about eps.
    """
    return cd_over_cl(CL, CD)