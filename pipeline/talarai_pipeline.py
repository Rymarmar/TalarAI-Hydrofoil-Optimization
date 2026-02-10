"""
Purpose:
  Define how a 6-parameter latent design becomes a physical airfoil coordinate set,
  and how we evaluate it using NeuralFoil (CL/CD).

Key flow:
  latent(6) → scale → decoder → y80 (upper40 + lower40)
  y80 + fixed x-grid → (x,y) coordinates
  coords → NeuralFoil → CL, CD
"""

from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
import neuralfoil as nf

# decoder model and scaler params are loaded from disk
BASE_DIR = Path(__file__).resolve().parent
DECODER_MODEL_PATH = BASE_DIR / "decoder_model_6x100x1000x80.h5"
SCALER_PATH = BASE_DIR / "scaler_params.npz"
# feeds the decoder latents in the same normalized scale it was trained on

#  helper that standardizes outputs into Python floats 
def _to_scalar(x):
    """Convert numpy scalars/arrays to a normal Python float."""
    if x is None:
        raise ValueError("Got None where a numeric value was expected.")
    if isinstance(x, (int, float)):
        return float(x)
    arr = np.asarray(x)
    if arr.ndim == 0:
        return float(arr)
    return float(arr.ravel()[0])


class TalarAIPipeline:
    def __init__(self, n_points: int = 40):
        # number of points per surface (upper 40 + lower 40 = 80 total y-values)
        self.n_points = int(n_points)

        # common x-grid from LE (0) to TE (1) (to match the decoder model and scaler arrays)
        self.x_common = np.linspace(0.0, 1.0, self.n_points).astype(np.float32)

        # load decoder and scaler
        if not DECODER_MODEL_PATH.exists():
            raise FileNotFoundError(f"Decoder model not found at: {DECODER_MODEL_PATH}")
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler params not found at: {SCALER_PATH}")

        self.decoder = load_model(str(DECODER_MODEL_PATH), compile=False)

        scaler_data = np.load(str(SCALER_PATH))
        self.scaler_mean = scaler_data["mean"].astype(np.float32)
        self.scaler_scale = scaler_data["scale"].astype(np.float32)
        self.scaler_scale = np.where(self.scaler_scale == 0, 1.0, self.scaler_scale)

    def latent_to_y80(self, latent_vec):
        """
        6D latent → normalized → decoder → y80

        y80 layout:
          first 40 values = upper surface y(x)
          next  40 values = lower surface y(x)
        """
        latent_vec = np.asarray(latent_vec, dtype=np.float32).reshape(1, -1)
        if latent_vec.shape[1] != 6:
            raise ValueError(f"Expected latent vector length 6, got {latent_vec.shape}")

        # match training normalization
        latent_scaled = (latent_vec - self.scaler_mean) / self.scaler_scale

        # decoder outputs shape (80,)
        y80 = self.decoder.predict(latent_scaled, verbose=0)[0].astype(np.float32)
        return y80

    def latent_to_coordinates(self, latent_vec):
        """
        Build foil coordinates in a single, consistent convention

        Convention used:
          - upper surface: x 0→1
          - lower surface: x 1→0 (reverse x)
          - lower y is reversed to match x reversal

        This produces one closed-ish loop around the airfoil
        """
        y80 = self.latent_to_y80(latent_vec)

        y_upper = y80[: self.n_points]
        y_lower = y80[self.n_points :]

        x_upper = self.x_common
        x_lower = self.x_common[::-1]
        y_lower_rev = y_lower[::-1]

        x_coords = np.concatenate([x_upper, x_lower])
        y_coords = np.concatenate([y_upper, y_lower_rev])

        coords = np.stack([x_coords, y_coords], axis=1).astype(np.float32)
        return coords

    # Two functions for evaluation steps
    def _eval_coords_neuralfoil(self, coords, alpha, Re, model_size):
        """
        Run NeuralFoil on coordinates at a single operating point (alpha, Re).
        """
        aero = nf.get_aero_from_coordinates(
            coordinates=coords,
            alpha=alpha,
            Re=Re,
            model_size=model_size,
        )
        out = dict(aero)
        out["coords"] = coords
        return out

    # Pass coordinates through NF with alpha and Re, and extract CL/CD for use later
    def eval_latent_with_neuralfoil(
        self,
        latent_vec,
        alpha=6.0,
        Re=5e5,
        model_size="xlarge",
    ):
        """
        Main evaluation function used by NOM:

          latent → coords → NeuralFoil → CL, CD

        We do NOT auto-flip/mirror to force positive CL anymore.
        If CL comes out negative, that indicates a real convention/shape issue
        that should be handled by constraints or data/representation.
        """
        coords = self.latent_to_coordinates(latent_vec)
        out = self._eval_coords_neuralfoil(coords, alpha, Re, model_size)

        out["CL"] = float(_to_scalar(out.get("CL")))
        out["CD"] = float(_to_scalar(out.get("CD")))
        out["coords"] = coords

        # for compatibility with older code
        out["fix_mode"] = "none"
        return out


if __name__ == "__main__":
    # quick smoke test
    pipeline = TalarAIPipeline()
    latent = np.zeros(6, dtype=np.float32)
    out = pipeline.eval_latent_with_neuralfoil(latent, alpha=6.0, Re=5e5)
    print("CL:", out["CL"], "CD:", out["CD"])
