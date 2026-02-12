"""
pipeline/talarai_pipeline.py

latent(6) -> decoder -> y80 (upper40 + lower40) -> coords(80x2) -> NeuralFoil -> CL, CD

Meeting fixes:
  - No scaler
  - Coordinate orientation matches .dat loop: 1 -> 0 -> 1
  - Optional debug prints
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
import neuralfoil as nf


BASE_DIR = Path(__file__).resolve().parent

CANDIDATE_DECODER_FILES = [
    BASE_DIR / "decoder_model_6x100x1000x80.weights.h5",
    BASE_DIR / "decoder_model_6x100x1000x80.h5",
]


def _to_float(x) -> float:
    arr = np.asarray(x)
    return float(arr.reshape(-1)[0])


class TalarAIPipeline:
    def __init__(self, n_points: int = 40, decoder_model_path: str | Path | None = None):
        self.n_points = int(n_points)
        if self.n_points <= 1:
            raise ValueError("n_points must be >= 2")

        # .dat loop convention:
        # upper surface TE->LE (x: 1 -> 0)
        # lower surface LE->TE (x: 0 -> 1)
        self.x_grid = np.linspace(0.0, 1.0, self.n_points).astype(np.float32)
        self.x_upper = self.x_grid[::-1]
        self.x_lower = self.x_grid

        # Load decoder model
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
        latent -> decoder -> y (2*n_points,)

        Layout from training:
          y[:n_points] = upper y values (stored in 0->1 direction)
          y[n_points:] = lower y values (stored in 0->1 direction)

        When making coords, we flip the UPPER y to match x_upper (1->0).
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
        coords shape: (2*n_points, 2)

        Upper: x = 1->0, y = reverse(y_upper)
        Lower: x = 0->1, y = y_lower
        """
        y = self.latent_to_y(latent_vec, debug=debug)
        y_upper = y[: self.n_points]
        y_lower = y[self.n_points :]

        y_upper_rev = y_upper[::-1]

        if debug:
            print("\n--- Coordinate Construction ---")
            print("x_upper (1→0):", self.x_upper[:5], "...", self.x_upper[-5:])
            print("y_upper (1→0):", y_upper_rev[:5], "...", y_upper_rev[-5:])
            print("x_lower (0→1):", self.x_lower[:5], "...", self.x_lower[-5:])
            print("y_lower (0→1):", y_lower[:5], "...", y_lower[-5:])

        x_coords = np.concatenate([self.x_upper, self.x_lower])
        y_coords = np.concatenate([y_upper_rev, y_lower])
        coords = np.stack([x_coords, y_coords], axis=1).astype(np.float32)

        if debug:
            print(f"Coordinates shape: {coords.shape} (should be {(2*self.n_points,2)})")

        return coords

    def eval_latent_with_neuralfoil(
        self,
        latent_vec: np.ndarray,
        alpha: float = 6.0,
        Re: float = 5e5,
        model_size: str = "xlarge",
        debug: bool = False,
    ) -> dict:
        coords = self.latent_to_coordinates(latent_vec, debug=debug)

        # NeuralFoil API notes (for meeting):
        #   nf.get_aero_from_coordinates(coordinates=Nx2, alpha=..., Re=...) returns a dict-like
        #   object with keys including: "CL", "CD", and "analysis_confidence".
        #   See NeuralFoil README for example usage and returned keys.
        aero = nf.get_aero_from_coordinates(
            coordinates=coords,
            alpha=float(alpha),
            Re=float(Re),
            model_size=model_size,
        )

        # Copy output so we don’t mutate whatever NeuralFoil returns.
        out = aero if isinstance(aero, dict) else dict(aero)
        out["coords"] = coords
        out["CL"] = _to_float(out.get("CL"))
        out["CD"] = _to_float(out.get("CD"))
        return out