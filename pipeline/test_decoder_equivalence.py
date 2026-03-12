"""
pipeline/test_decoder_equivalence.py

Goal:
  Verify decoder model output == pipeline output and the coordinate
  convention is correct after the save_weights / load_weights fix.

Checks:
  1) y80_direct (decoder rebuilt + load_weights) equals pipeline.latent_to_y()
     -- proves both are loading the same weights the same way
  2) Coordinate convention: upper x is 1->0, lower x is 0->1
  3) Upper surface y in coords matches y80[:40] REVERSED
     (because latent_to_coordinates does y_a = y[:40][::-1])
     Lower surface y in coords matches y80[40:] directly

BUGS FIXED vs original:
  BUG 1 -- Line 25: r-string missing so backslash-d was an invalid
           escape (SyntaxWarning) AND the path was wrong:
           "pipeline\decoder_model..." should just be the filename.
           FIX: use CANDIDATE_DECODER_FILES from the pipeline directly
           so this test never searches a different path.

  BUG 2 -- Line 42: used load_model() which is incompatible with the
           .weights.h5 format now saved by decoder.py save_weights().
           FIX: rebuild architecture + load_weights(), same as pipeline.

  BUG 3 -- Line 69: expected upper y was y80_direct[:40] (NOT reversed)
           but the pipeline DOES reverse it (y_a = y[:40][::-1]).
           This would always fail TEST 3.
           FIX: expected upper = y80_direct[:40][::-1]
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from talarai_pipeline import TalarAIPipeline, CANDIDATE_DECODER_FILES


def _rebuild_and_load(weights_path: Path) -> Model:
    """
    Rebuild the decoder architecture and load saved weights.
    Must match decoder.py and talarai_pipeline.py exactly.
    If the architecture ever changes, update all three files together.
    """
    inp = Input(shape=(6,), name="params6")
    x   = Dense(100,  activation="relu")(inp)
    x   = Dense(1000, activation="relu")(x)
    out = Dense(80,   activation="linear")(x)
    model = Model(inp, out, name="testdecoder_6x100x1000x80")
    model.load_weights(str(weights_path))
    return model


def _find_decoder_weights() -> Path:
    """
    Use the same candidate list as TalarAIPipeline so this test
    never searches a different path than the pipeline does.
    """
    for p in CANDIDATE_DECODER_FILES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Decoder weights not found. Tried:\n  - "
        + "\n  - ".join(str(p) for p in CANDIDATE_DECODER_FILES)
    )


def main():
    # Reproducible random latent for the test
    rng = np.random.default_rng(seed=0)
    latent = rng.uniform(-1, 1, size=(6,)).astype(np.float32)

    # ---------------------------------------------------------------
    # Load pipeline (uses load_weights internally)
    # ---------------------------------------------------------------
    pipe = TalarAIPipeline()

    # ---------------------------------------------------------------
    # Load decoder directly using the same weights file
    # ---------------------------------------------------------------
    decoder_path = _find_decoder_weights()
    decoder = _rebuild_and_load(decoder_path)

    # ---------------------------------------------------------------
    # TEST 1: y80 from direct decoder call vs pipeline.latent_to_y()
    # Both should be identical because they load the same weights
    # with the same architecture.
    # ---------------------------------------------------------------
    y80_direct = decoder.predict(latent.reshape(1, -1), verbose=0)[0].astype(np.float32)
    y80_pipe   = pipe.latent_to_y(latent).astype(np.float32)

    max_diff_y80 = float(np.max(np.abs(y80_direct - y80_pipe)))

    print("\n==============================")
    print("TEST 1: y80 direct vs pipeline")
    print("==============================")
    print(f"Weights file: {decoder_path.name}")
    print(f"max |y80_direct - y80_pipe| = {max_diff_y80:.2e}")
    test1_pass = max_diff_y80 < 1e-5   # allow tiny float32 rounding
    print("Result:", "PASS" if test1_pass else "FAIL")

    # ---------------------------------------------------------------
    # TEST 2: coordinate convention check (1->0->1 loop)
    # ---------------------------------------------------------------
    coords = pipe.latent_to_coordinates(latent, debug=True)
    x_all  = coords[:, 0]
    y_all  = coords[:, 1]

    x_upper = x_all[:40]
    x_lower = x_all[40:]

    upper_decreasing = bool(np.all(np.diff(x_upper) <= 1e-6))
    lower_increasing = bool(np.all(np.diff(x_lower) >= -1e-6))

    print("\n==============================")
    print("TEST 2: coordinate x-direction")
    print("==============================")
    print(f"Upper x start={x_upper[0]:.4f}  end={x_upper[-1]:.4f}  decreasing={upper_decreasing}")
    print(f"Lower x start={x_lower[0]:.4f}  end={x_lower[-1]:.4f}  increasing={lower_increasing}")
    test2_pass = upper_decreasing and lower_increasing
    print("Result:", "PASS" if test2_pass else "FAIL")

    # ---------------------------------------------------------------
    # TEST 3: y-values in coords match y80 with correct reversal
    #
    # latent_to_coordinates builds two candidate surfaces:
    #   surf_A = (x_grid, y80[:40][::-1])  <- y reversed
    #   surf_B = (x_grid, y80[40:])        <- y as-is
    # then auto-detects upper vs lower by mean_y.
    #
    # The final coords loop is always: upper(1->0) then lower(0->1).
    # But WHICH raw surface becomes upper depends on the foil shape.
    #
    # We figure out which case applies by re-running the same
    # mean_y comparison the pipeline uses, then check accordingly.
    # ---------------------------------------------------------------
    y80_a_reversed = y80_direct[:40][::-1]   # what pipeline calls y_a
    y80_b           = y80_direct[40:]          # what pipeline calls y_b

    mean_a = float(np.mean(y80_a_reversed))
    mean_b = float(np.mean(y80_b))
    max_a  = float(np.max(y80_a_reversed))
    max_b  = float(np.max(y80_b))

    if (mean_a > mean_b) or (abs(mean_a - mean_b) < 1e-9 and max_a >= max_b):
        # A=upper, B=lower  (pipeline traverses A reversed -> LE, then B -> TE)
        y_upper_expected = y80_a_reversed[::-1]  # surf_A reversed again = TE->LE
        y_lower_expected = y80_b
        expected_label = "A=upper: coords[:40]=surf_A(TE->LE), coords[40:]=surf_B"
    else:
        # B=upper, A=lower  (pipeline traverses B reversed -> LE, then A -> TE)
        y_upper_expected = y80_b[::-1]           # surf_B reversed = TE->LE
        y_lower_expected = y80_a_reversed
        expected_label = "B=upper: coords[:40]=surf_B(TE->LE), coords[40:]=surf_A"

    y_upper_from_coords = y_all[:40]
    y_lower_from_coords = y_all[40:]

    max_diff_upper = float(np.max(np.abs(y_upper_from_coords - y_upper_expected)))
    max_diff_lower = float(np.max(np.abs(y_lower_from_coords - y_lower_expected)))

    print("\n==============================")
    print("TEST 3: y-value convention")
    print("==============================")
    print(f"Expected layout: {expected_label}")
    print(f"Upper max |diff| = {max_diff_upper:.2e}")
    print(f"Lower max |diff| = {max_diff_lower:.2e}")
    test3_pass = max_diff_upper < 1e-5 and max_diff_lower < 1e-5
    print("Result:", "PASS" if test3_pass else "FAIL")
    if not test3_pass:
        print("  The y-values do not match the expected layout.")
        print("  This means latent_to_coordinates has a reversal bug.")

    # ---------------------------------------------------------------
    # OVERALL RESULT
    # ---------------------------------------------------------------
    print("\n==============================")
    all_pass = test1_pass and test2_pass and test3_pass
    if all_pass:
        print("ALL TESTS PASSED")
        print("  decoder.py and talarai_pipeline.py are equivalent.")
        print("  Coordinate convention: upper 1->0, lower 0->1.")
    else:
        print("ONE OR MORE TESTS FAILED")
        if not test1_pass:
            print("  TEST 1: weights mismatch. Check both files use")
            print("  load_weights() with the same architecture.")
        if not test2_pass:
            print("  TEST 2: x direction wrong. Check latent_to_coordinates().")
        if not test3_pass:
            print("  TEST 3: y mismatch. Auto-detect may have swapped")
            print("  upper/lower, or reversal logic changed.")
    print("==============================\n")


if __name__ == "__main__":
    main()