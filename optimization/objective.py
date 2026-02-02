import numpy as np


def cd_over_abscl(CL: float, CD: float, eps: float = 1e-9) -> float:
    """
    Sponsor-safe objective: minimize CD/|CL|.
    Equivalent to maximizing |CL|/CD.

    Why abs(CL)?
    - Your pipeline currently has sign convention issues.
    - This keeps optimization meaningful while geometry convention is standardized later.
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
    Default objective used by NOM.
    Change this ONE function if your professor wants CD/CL instead of CD/|CL|.
    """
    return cd_over_abscl(CL, CD)
