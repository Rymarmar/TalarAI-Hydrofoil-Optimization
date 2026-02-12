"""
pipeline/test_decoder_equivalence.py

Goal:
  Verify decoder model output == pipeline output and the coordinate convention is correct.

Checks:
  1) y80_direct (decoder.predict) equals pipeline.latent_to_y80
  2) coords produced by pipeline match y80 meaning:
       upper coords y == reverse(y80[:40])
       lower coords y == y80[40:]
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model

from talarai_pipeline import TalarAIPipeline


def _find_decoder_model(pipeline_dir: Path) -> Path:
    candidates = [
        pipeline_dir / "decoder_model_6x100x1000x80.weights.h5",
        pipeline_dir / "decoder_model_6x100x1000x80.h5",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Decoder model not found. Tried:\n  - " + "\n  - ".join(str(c) for c in candidates)
    )


def main():
    pipeline_dir = Path(__file__).resolve().parent

    # 1) load pipeline + direct decoder
    pipe = TalarAIPipeline()
    decoder_path = _find_decoder_model(pipeline_dir)
    decoder = load_model(str(decoder_path), compile=False)

    # 2) test latent (random is fine; you can swap in a dataset latent later)
    latent = np.random.uniform(-1, 1, size=(6,)).astype(np.float32)

    # 3) y80 direct vs pipeline
    y80_direct = decoder.predict(latent.reshape(1, -1), verbose=0)[0].astype(np.float32)
    y80_pipe = pipe.latent_to_y(latent).astype(np.float32)

    max_diff_y80 = float(np.max(np.abs(y80_direct - y80_pipe)))

    print("\n==============================")
    print("TEST 1: y80 direct vs pipeline")
    print("==============================")
    print("decoder_path:", decoder_path.name)
    print("max |y80_direct - y80_pipe| =", max_diff_y80)

    # 4) coords consistency check (1->0->1)
    coords = pipe.latent_to_coordinates(latent, debug=True)
    x = coords[:, 0]
    y = coords[:, 1]

    x_upper = x[:40]
    x_lower = x[40:]
    upper_decreasing = bool(np.all(np.diff(x_upper) <= 1e-8))
    lower_increasing = bool(np.all(np.diff(x_lower) >= -1e-8))

    y_upper_expected = y80_direct[:40][::-1]
    y_lower_expected = y80_direct[40:]

    y_upper_from_coords = y[:40]
    y_lower_from_coords = y[40:]

    max_diff_upper = float(np.max(np.abs(y_upper_from_coords - y_upper_expected)))
    max_diff_lower = float(np.max(np.abs(y_lower_from_coords - y_lower_expected)))

    print("\n==============================")
    print("TEST 2: coordinate convention")
    print("==============================")
    print("Upper x decreasing (1->0)?", upper_decreasing)
    print("Lower x increasing (0->1)?", lower_increasing)
    print("Upper max |diff| (coords vs expected):", max_diff_upper)
    print("Lower max |diff| (coords vs expected):", max_diff_lower)

    print("\n==============================")
    if (
        max_diff_y80 < 1e-6
        and upper_decreasing
        and lower_increasing
        and max_diff_upper < 1e-6
        and max_diff_lower < 1e-6
    ):
        print("✅ PASS: decoder.py model + talarai_pipeline.py are equivalent and use 1→0→1.")
    else:
        print("❌ FAIL: mismatch detected.")
        print("   If TEST 1 fails: pipeline isn't using same decoder model or latent preprocessing differs.")
        print("   If TEST 2 fails: latent_to_coordinates convention is wrong (upper/lower reversal mismatch).")
    print("==============================\n")


if __name__ == "__main__":
    main()
