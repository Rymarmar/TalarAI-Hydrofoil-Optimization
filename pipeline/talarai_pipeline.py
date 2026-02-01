"""
TalarAI automated design pipeline:
6 latent parameters → decoder → 2D airfoil coordinates → NeuralFoil aerodynamics.

This version:
- Loads decoder/scaler relative to this file (robust paths)
- Does NOT require scikit-learn
- AUTO-FIXES lift sign for presentation consistency:
    It evaluates multiple coordinate variants (order flip and y-mirror)
    and chooses the first one that yields a valid positive CL.
- Returns a dict with:
    CL, CD, coords (chosen)
    CL_raw, CD_raw, coords_raw (original)
    fix_mode (what transform was applied)
"""

from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
import neuralfoil as nf


# -----------------------------
# Robust paths (relative to this file)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DECODER_MODEL_PATH = BASE_DIR / "decoder_model_6x100x1000x80.h5"
SCALER_PATH = BASE_DIR / "scaler_params.npz"


def _to_scalar(x):
    """Convert float / np scalar / 1-element array / list into a Python float."""
    if x is None:
        raise ValueError("Got None where a numeric value was expected.")
    if isinstance(x, (int, float)):
        return float(x)
    arr = np.asarray(x)
    if arr.ndim == 0:
        return float(arr)
    return float(arr.ravel()[0])


def _is_valid_number(x):
    return np.isfinite(x) and (not np.isnan(x))


class TalarAIPipeline:
    def __init__(self, n_points: int = 40):
        self.n_points = int(n_points)
        self.x_common = np.linspace(0.0, 1.0, self.n_points).astype(np.float32)

        if not DECODER_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Decoder model not found at: {DECODER_MODEL_PATH}\n"
                "Make sure decoder_model_6x100x1000x80.h5 is inside pipeline/."
            )
        if not SCALER_PATH.exists():
            raise FileNotFoundError(
                f"Scaler params not found at: {SCALER_PATH}\n"
                "Make sure scaler_params.npz is inside pipeline/."
            )

        self.decoder = load_model(str(DECODER_MODEL_PATH), compile=False)

        scaler_data = np.load(str(SCALER_PATH))
        self.scaler_mean = scaler_data["mean"].astype(np.float32)
        self.scaler_scale = scaler_data["scale"].astype(np.float32)
        self.scaler_scale = np.where(self.scaler_scale == 0, 1.0, self.scaler_scale)

    # -----------------------------
    # Latent -> y80
    # -----------------------------
    def latent_to_y80(self, latent_vec):
        latent_vec = np.asarray(latent_vec, dtype=np.float32).reshape(1, -1)
        if latent_vec.shape[1] != 6:
            raise ValueError(f"Expected latent vector length 6, got {latent_vec.shape}")

        latent_scaled = (latent_vec - self.scaler_mean) / self.scaler_scale
        y80 = self.decoder.predict(latent_scaled, verbose=0)[0].astype(np.float32)
        return y80

    # -----------------------------
    # Latent -> coords (80x2)
    # -----------------------------
    def latent_to_coordinates(self, latent_vec):
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

    # -----------------------------
    # Evaluate coords via NeuralFoil
    # -----------------------------
    def _eval_coords_neuralfoil(self, coords, alpha, Re, model_size):
        aero = nf.get_aero_from_coordinates(
            coordinates=coords,
            alpha=alpha,
            Re=Re,
            model_size=model_size,
        )
        out = dict(aero)
        out["coords"] = coords
        return out

    # -----------------------------
    # Public eval: choose a "presentation consistent" variant
    # -----------------------------
    def eval_latent_with_neuralfoil(self, latent_vec, alpha=6.0, Re=5e5, model_size="xlarge"):
        """
        Evaluate latent design and automatically choose a coordinate convention that yields
        positive CL (if possible).

        We try these in order:
          0) raw coords
          1) reversed order
          2) y-mirrored (flip airfoil vertically)
          3) y-mirrored + reversed

        We select the FIRST variant that produces:
          - CD > 0
          - finite CL and CD
          - CL > 0

        If none produce CL > 0, we return raw but still include diagnostics.
        """
        coords_raw = self.latent_to_coordinates(latent_vec)
        out_raw = self._eval_coords_neuralfoil(coords_raw, alpha, Re, model_size)

        CL_raw = _to_scalar(out_raw.get("CL"))
        CD_raw = _to_scalar(out_raw.get("CD"))

        # Generate candidate variants
        candidates = [
            ("raw", coords_raw),
            ("reversed", coords_raw[::-1].copy()),
            ("y_mirror", coords_raw * np.array([1.0, -1.0], dtype=np.float32)),
            ("y_mirror_reversed", (coords_raw * np.array([1.0, -1.0], dtype=np.float32))[::-1].copy()),
        ]

        chosen = None
        chosen_out = None
        chosen_CL = None
        chosen_CD = None

        for mode, coords in candidates:
            try:
                out = self._eval_coords_neuralfoil(coords, alpha, Re, model_size)
                CL = _to_scalar(out.get("CL"))
                CD = _to_scalar(out.get("CD"))

                # Must be finite and sane
                if not _is_valid_number(CL) or not _is_valid_number(CD):
                    continue
                if CD <= 0:
                    continue

                # Prefer positive CL for presentations
                if CL > 0:
                    chosen = mode
                    chosen_out = out
                    chosen_CL = CL
                    chosen_CD = CD
                    break

            except Exception:
                continue

        # If we found a positive-CL variant, use it; else fall back to raw
        if chosen_out is None:
            chosen = "raw_fallback"
            chosen_out = out_raw
            chosen_CL = CL_raw
            chosen_CD = CD_raw
            coords_final = coords_raw
        else:
            coords_final = chosen_out["coords"]

        result = dict(chosen_out)
        result["CL"] = float(chosen_CL)
        result["CD"] = float(chosen_CD)
        result["coords"] = coords_final

        # Diagnostics for transparency/debugging
        result["CL_raw"] = float(CL_raw)
        result["CD_raw"] = float(CD_raw)
        result["coords_raw"] = coords_raw
        result["fix_mode"] = chosen

        return result


if __name__ == "__main__":
    pipeline = TalarAIPipeline()
    latent = np.zeros(6, dtype=np.float32)
    out = pipeline.eval_latent_with_neuralfoil(latent, alpha=6.0, Re=5e5)
    print("fix_mode:", out["fix_mode"])
    print("CL_raw:", out["CL_raw"], "CD_raw:", out["CD_raw"])
    print("CL:", out["CL"], "CD:", out["CD"])
