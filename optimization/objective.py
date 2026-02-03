import numpy as np


def cd_over_abscl(CL: float, CD: float, eps: float = 1e-9) -> float:
    """
    Sponsor-safe objective: minimize CD/|CL|.
    Equivalent to maximizing |CL|/CD.

    Why abs(CL)?
    - Your pipeline can still produce negative CL depending on coord orientation.
    - This keeps optimization meaningful until sign convention is standardized.
    """
    return float(CD) / max(float(abs(CL)), eps)


def cd_over_cl(CL: float, CD: float, eps: float = 1e-9) -> float:
    """
    Strict objective: minimize CD/CL.
    Equivalent to maximizing CL/CD.
    ONLY use once CL sign convention is consistently positive.
    """
    return float(CD) / max(float(CL), eps)


def default_objective(CL: float, CD: float) -> float:
    """
    Default objective used everywhere.
    Change this ONE function if professor wants CD/CL instead of CD/|CL|.
    """
    return cd_over_abscl(CL, CD)
