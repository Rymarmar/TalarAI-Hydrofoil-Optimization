"""
pipeline/talarai_pipeline.py

latent(6) -> decoder -> y80 (two surfaces) -> coords(80x2) -> NeuralFoil -> CL, CD

KEY FIX (your “infinity shape” problem):
- We DO NOT assume y[:40] is upper and y[40:] is lower.
- Instead, we build both surfaces in x=0->1, then AUTO-DETECT:
    upper = the surface with higher mean y (or higher max y)
    lower = the other one

Then we assemble the final .dat loop for NeuralFoil:
    upper: TE -> LE (x 1->0)   [we reverse the ROWS]
    lower: LE -> TE (x 0->1)   [keep as-is]
Overall loop: 1 -> 0 -> 1

This avoids “reversing logic” as a hack — we only reverse the final traversal
direction, which is required to build a closed loop in the standard airfoil format.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
import neuralfoil as nf


BASE_DIR = Path(__file__).resolve().parent

CANDIDATE_DECODER_FILES = [
    BASE_DIR / "decoder_model_6x100x1000x80.weights.h5",
]


def _to_float(x) -> float:
    arr = np.asarray(x)
    return float(arr.reshape(-1)[0])


def _prep_coords_for_neuralfoil(coords: np.ndarray, n_points: int = 40) -> np.ndarray:
    """
    Clean up decoded coordinates so NeuralFoil's Kulfan fitter doesn't crash.

    PROBLEM:
      The decoder outputs 80 points: 40 upper (TE->LE) + 40 lower (LE->TE).
      Both surfaces use x_grid = linspace(0, 1, 40), so:
        - Row 39 (upper end) and row 40 (lower start) both have x=0.0
          → duplicate leading edge point
        - Exact x=0.0 and x=1.0 create degenerate rows in the Kulfan
          basis C(x) = x^0.5 * (1-x)^1.0 (evaluates to exactly 0)
        - Open trailing edge (upper TE y ≠ lower TE y)

      These cause numpy.linalg.lstsq to fail with "SVD did not converge",
      which shows up as the "DLASCLS parameter 4" error spam.

    FIX:
      1) Remove the duplicate LE point (drop first point of lower surface)
      2) Nudge x values: clamp to [1e-6, 1-1e-6] so basis functions
         never hit exactly 0
      3) Close the trailing edge: average upper/lower TE y-values so
         the airfoil loop closes properly

    This produces a clean 79-point airfoil loop (TE -> LE -> TE) that
    AeroSandbox's Kulfan fitter can handle reliably.

    INPUT:  coords shape (80, 2) — original decoder output
    OUTPUT: cleaned coords shape (79, 2) — ready for NeuralFoil
    """
    coords = np.array(coords, dtype=np.float64)

    upper = coords[:n_points]       # rows 0-39: TE->LE (x: 1->0)
    lower = coords[n_points:]       # rows 40-79: LE->TE (x: 0->1)

    # 1) Remove duplicate LE point: upper ends at x=0, lower starts at x=0.
    #    Drop the first row of lower so the LE appears only once.
    lower_no_dup = lower[1:]        # rows 41-79 (39 points)

    # 2) Close trailing edge: average upper TE (row 0) and lower TE (last row)
    te_y_avg = (upper[0, 1] + lower_no_dup[-1, 1]) / 2.0
    upper[0, 1] = te_y_avg
    lower_no_dup[-1, 1] = te_y_avg

    # 3) Reassemble the loop
    clean = np.vstack([upper, lower_no_dup])  # 40 + 39 = 79 points

    # 4) Nudge x away from exact 0 and 1 to avoid degenerate Kulfan basis
    clean[:, 0] = np.clip(clean[:, 0], 1e-6, 1.0 - 1e-6)

    return clean.astype(np.float64)


class TalarAIPipeline:
    """
    Owns:
      - x_grid definition (same grid used during training)
      - decoder model (loaded once)
    Provides:
      - latent_to_y()
      - latent_to_coordinates()
      - eval_latent_with_neuralfoil()
    """

    def __init__(self, n_points: int = 40, decoder_model_path: str | Path | None = None):
        self.n_points = int(n_points)
        if self.n_points < 2:
            raise ValueError("n_points must be >= 2")

        # Training representation uses x in 0 -> 1 order
        self.x_grid = np.linspace(0.0, 1.0, self.n_points).astype(np.float32)

        # Find/load decoder
        if decoder_model_path is not None:
            model_path = Path(decoder_model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Decoder model not found at: {model_path}")
            found = model_path
        else:
            found = None
            for p in CANDIDATE_DECODER_FILES:
                if p.exists():
                    found = p
                    break
            if found is None:
                raise FileNotFoundError(
                    "Decoder model not found. Tried:\n  - " + "\n  - ".join(str(p) for p in CANDIDATE_DECODER_FILES)
                )

        self.decoder_path = found
        self.decoder = load_model(str(found), compile=False)

    def latent_to_y(self, latent_vec: np.ndarray, debug: bool = False) -> np.ndarray:
        """
        latent (6,) -> decoder output y (2*n_points,)

        IMPORTANT:
        - Decoder returns TWO surfaces in x=0->1 order.
        - But which half is “upper vs lower” is NOT assumed here anymore.
        """
        z = np.asarray(latent_vec, dtype=np.float32).reshape(1, -1)
        if z.shape[1] != 6:
            raise ValueError(f"Expected latent length 6, got {z.shape}")

        y = self.decoder.predict(z, verbose=0)[0].astype(np.float32)

        expected = (2 * self.n_points,)
        if y.shape != expected:
            raise ValueError(f"Decoder returned {y.shape}, expected {expected}")

        if debug:
            print("Decoder output length:", len(y))

        return y

    def latent_to_coordinates(self, latent_vec: np.ndarray, debug: bool = False) -> np.ndarray:
        """
        Convert latent -> y80 -> coords (80 x 2) with correct loop:

          1) Build surfA and surfB in natural x=0->1
          2) Decide which is upper vs lower (auto-detect)
          3) Assemble final loop:
               upper TE->LE (reverse rows)
               lower LE->TE (keep rows)
        """
        y = self.latent_to_y(latent_vec, debug=debug)

        # Two candidate surfaces (both are x=0->1)
        # y_a = y[: self.n_points] THIS IS WHAT CAUSES LOWER TO BE FLIPPED
        y_a = y[: self.n_points][::-1]
        y_b = y[self.n_points :] 
        

        surf_a_01 = np.column_stack([self.x_grid, y_a]).astype(np.float32)
        surf_b_01 = np.column_stack([self.x_grid, y_b]).astype(np.float32)

        # --- AUTO-DETECT upper vs lower ---
        # Use mean y as the primary classifier; fall back to max y if needed.
        mean_a = float(np.mean(surf_a_01[:, 1]))
        mean_b = float(np.mean(surf_b_01[:, 1]))
        max_a = float(np.max(surf_a_01[:, 1]))
        max_b = float(np.max(surf_b_01[:, 1]))

        if (mean_a > mean_b) or (abs(mean_a - mean_b) < 1e-9 and max_a >= max_b):
            upper_01 = surf_a_01
            lower_01 = surf_b_01
            detected = "A=upper, B=lower"
        else:
            upper_01 = surf_b_01
            lower_01 = surf_a_01
            detected = "B=upper, A=lower"

        # Final traversal loop for .dat / NeuralFoil:
        upper_te_to_le = upper_01[::-1]   # x 1->0
        lower_le_to_te = lower_01         # x 0->1
        coords = np.vstack([upper_te_to_le, lower_le_to_te]).astype(np.float32)

        if debug:
            print("\n--- 10-second orientation diagnostic ---")
            print("Detected:", detected)
            print(f"mean_y(A)={mean_a:.6f}  max_y(A)={max_a:.6f}")
            print(f"mean_y(B)={mean_b:.6f}  max_y(B)={max_b:.6f}")
            print("Goal loop: 1 -> 0 -> 1")
            print("Upper x start/end:", float(coords[0, 0]), float(coords[self.n_points - 1, 0]))
            print("Lower x start/end:", float(coords[self.n_points, 0]), float(coords[-1, 0]))
            print("Coords shape:", coords.shape)

        return coords

    def eval_latent_with_neuralfoil(
        self,
        latent_vec: np.ndarray,
        alpha: float = 6.0,
        Re: float = 5e5,
        model_size: str = "xlarge",
        debug: bool = False,
    ) -> dict:
        """
        Evaluate with NeuralFoil:
          nf.get_aero_from_coordinates(coordinates=coords, alpha=..., Re=...)
        """
        coords = self.latent_to_coordinates(latent_vec, debug=debug)

        # -----------------------------------------------------------------
        # FIX (3/9/26): Clean up coordinates before NeuralFoil.
        #
        # NeuralFoil internally converts coordinates to Kulfan parameters
        # using a least-squares fit with basis functions:
        #   C(x) = x^0.5 * (1-x)^1.0
        #
        # This fails (SVD did not converge / DLASCLS error) when:
        #   1) Duplicate points at LE (x=0 appears twice — upper end
        #      and lower start) create rank-deficient rows in the matrix
        #   2) Exact x=0.0 or x=1.0 make the basis function evaluate
        #      to exactly 0, creating degenerate rows
        #
        # The fix:
        #   a) Remove the duplicate LE point (drop first row of lower)
        #   b) Clamp x slightly away from exact 0 and 1
        #   c) Close the TE by averaging upper/lower TE y-values
        #
        # This does NOT change `coords` (which is used for geometry
        # checks in constraints.py). It only affects what NeuralFoil sees.
        # -----------------------------------------------------------------
        nf_coords = _prep_coords_for_neuralfoil(coords, self.n_points)

        aero = nf.get_aero_from_coordinates(
            coordinates=nf_coords,
            alpha=float(alpha),
            Re=float(Re),
            model_size=model_size,
        )

        out = aero if isinstance(aero, dict) else dict(aero)
        out["coords"] = coords  # return original coords (not cleaned)
        out["CL"] = _to_float(out.get("CL"))
        out["CD"] = _to_float(out.get("CD"))
        return out