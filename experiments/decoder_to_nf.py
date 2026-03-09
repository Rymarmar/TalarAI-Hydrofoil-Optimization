"""
experiments/decoder_to_nf.py

End-to-end test: latent params -> decoder -> coordinates -> NeuralFoil -> CL, CD
Compares decoded foil against original .dat file to verify decoder fidelity.

WHAT THIS TESTS:
  If the decoded NACA 2410 gives similar CL/CD to the real NACA 2410,
  the autoencoder is working well. If they diverge significantly,
  something is wrong with the encoder/decoder training.

FIXES (3/9/26):
  - Fixed backslash path causing SyntaxError (unicodeescape)
  - Fixed load_model() on .weights.h5 file (weights-only files need
    the architecture built first, then load_weights())
  - Removed unused StandardScaler import (scaler was removed from pipeline)
  - Updated CSV path to match current project structure
  - Uses TalarAIPipeline for decoding (same path NOM uses, so this
    test validates the exact same pipeline the optimizer runs)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

# Add project root to path so imports work from experiments/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import aerosandbox as asb
import neuralfoil as nf_lib
from pipeline.talarai_pipeline import TalarAIPipeline


# ============================================================
# PART 1 — LOAD LATENT PARAMS + DECODE AIRFOIL
# ============================================================

# Load latent parameters
# NOTE: your old script referenced "outputs/all_latent_params_numeric.csv"
# which doesn't exist. The actual file is in data/:
latent_csv = PROJECT_ROOT / "data" / "airfoil_latent_params_6.csv"
df = pd.read_csv(latent_csv)
params6 = df[['p1','p2','p3','p4','p5','p6']].values.astype(np.float32)
print(f"Loaded {len(params6)} foils from {latent_csv.name}")

# Load pipeline (handles decoder architecture + weights automatically)
# This is the SAME pipeline NOM uses, so if this test passes,
# the optimizer's decoder is also working correctly.
pipeline = TalarAIPipeline()
print(f"Pipeline ready: {pipeline.decoder_path.name}")

# Find target airfoil
target_name = "naca2410"
matches = df[df["filename"].str.contains(target_name, case=False)]
if len(matches) == 0:
    raise ValueError(f"No {target_name} in dataset!")
i = matches.index[0]
print(f"Found {target_name} at index {i} ({df['filename'].iloc[i]})")

# Decode: latent -> 80 coordinates (using the pipeline, same as NOM)
coords = pipeline.latent_to_coordinates(params6[i])
print(f"Decoded coordinates shape: {coords.shape}")

# Save predicted airfoil to CSV
output_csv = PROJECT_ROOT / "experiments" / f"{target_name}_predicted_airfoil.csv"
np.savetxt(output_csv, coords, delimiter=",", header="x,y", comments="")
print(f"Saved predicted foil: {output_csv}")

# Load original airfoil (.dat file from UIUC database)
orig_dat = PROJECT_ROOT / "data" / "coord_seligFmt" / f"{target_name}.dat"
if not orig_dat.exists():
    raise FileNotFoundError(f"Original .dat not found: {orig_dat}")
orig_data = np.loadtxt(orig_dat, skiprows=1)
print(f"Loaded original {target_name}: {orig_data.shape[0]} points")


# ============================================================
# PART 2 — AEROSANDBOX + NEURALFOIL SETUP
# ============================================================

# Original airfoil (full .dat resolution — uses AeroSandbox Kulfan fitting)
airfoil_orig = asb.Airfoil(name=f"{target_name}_orig", coordinates=orig_data)

# Decoded airfoil will use get_aero_from_coordinates directly (no Kulfan)
# because the 80-point decoded shape can confuse the Kulfan curve fitter


# ============================================================
# FLOW CONDITIONS
# ============================================================

alphas = np.linspace(-10, 15, 30)
velocity = 5.0     # m/s
chord = 5.0        # m (not real chord -- just for Re calculation)
rho = 1000          # kg/m^3 (water)
mu = 1e-3           # Pa*s (water dynamic viscosity)
Re_water = rho * velocity * chord / mu  # = 25,000,000

print(f"\nFlow conditions: Re = {Re_water:.0e}, alpha = {alphas[0]:.0f} to {alphas[-1]:.0f} deg")


def compute_cl_cd_airfoil(airfoil):
    """Compute CL/CD using get_aero_from_airfoil (for original .dat files).
    This uses AeroSandbox's Kulfan parameterization internally,
    which works well with clean, high-resolution .dat coordinates."""
    out = nf_lib.get_aero_from_airfoil(
        airfoil,
        alpha=alphas,
        Re=Re_water,
        model_size="large"
    )
    CL = np.asarray(out["CL"]).flatten()
    CD = np.asarray(out["CD"]).flatten()
    return CL, CD


def compute_cl_cd_pipeline(pipeline_obj, latent_vec, Re):
    """Compute CL/CD using the TalarAI pipeline — the EXACT same path
    NOM uses during optimization. This goes through:
      latent -> decoder -> coords -> NeuralFoil
    If this fails, NOM would fail too."""
    CL_list, CD_list = [], []
    for alpha in alphas:
        try:
            out = pipeline_obj.eval_latent_with_neuralfoil(
                latent_vec,
                alpha=float(alpha),
                Re=float(Re),
                model_size="xlarge",  # same as NOM default
            )
            CL_list.append(float(out["CL"]))
            CD_list.append(float(out["CD"]))
        except Exception as e:
            print(f"    FAILED at alpha={alpha:.1f}: {e}")
            CL_list.append(float("nan"))
            CD_list.append(float("nan"))
    return np.array(CL_list), np.array(CD_list)


# --- Coordinate diagnostic (helps debug decoder issues) ---
print("\n--- DECODED COORDINATE DIAGNOSTIC ---")
upper_te2le = coords[:40]
lower_le2te = coords[40:]
upper_le2te = upper_te2le[::-1]  # LE->TE
print(f"Upper x range: [{upper_le2te[0,0]:.4f}, {upper_le2te[-1,0]:.4f}]  (should be ~0 to ~1)")
print(f"Lower x range: [{lower_le2te[0,0]:.4f}, {lower_le2te[-1,0]:.4f}]  (should be ~0 to ~1)")
print(f"Upper y range: [{upper_le2te[:,1].min():.4f}, {upper_le2te[:,1].max():.4f}]")
print(f"Lower y range: [{lower_le2te[:,1].min():.4f}, {lower_le2te[:,1].max():.4f}]")
print(f"Max |y|: {np.max(np.abs(coords[:,1])):.4f}  (limit: 0.1964)")
print(f"Any NaN: {np.any(np.isnan(coords))}")
thickness = upper_le2te[:,1] - lower_le2te[:,1]
print(f"Thickness range: [{thickness.min():.4f}, {thickness.max():.4f}]")
if thickness.min() < 0:
    print(f"  WARNING: surfaces crossing at {np.sum(thickness < 0)} points!")
print("--- END DIAGNOSTIC ---\n")

print("Computing aero for original...")
CL_orig, CD_orig = compute_cl_cd_airfoil(airfoil_orig)
print("Computing aero for decoded (via pipeline, same as NOM)...")
CL_pred, CD_pred = compute_cl_cd_pipeline(pipeline, params6[i], Re_water)

# Print zero-AoA comparison
zi = np.argmin(np.abs(alphas))
n_valid = np.sum(np.isfinite(CL_pred))
print(f"\n{'='*50}")
print(f"DECODER EVAL: {n_valid}/{len(alphas)} alphas succeeded")
if n_valid == 0:
    print("ALL ALPHAS FAILED — decoder is producing bad coordinates.")
    print("You likely need to re-run encoder.py to regenerate latent params")
    print("that match your current decoder weights.")
    print(f"{'='*50}")
else:
    print(f"COMPARISON AT AoA = {alphas[zi]:.1f} deg:")
    if np.isfinite(CL_pred[zi]):
        print(f"  Original:  CL={CL_orig[zi]:.4f}  CD={CD_orig[zi]:.6f}  L/D={CL_orig[zi]/CD_orig[zi]:.1f}")
        print(f"  Decoded:   CL={CL_pred[zi]:.4f}  CD={CD_pred[zi]:.6f}  L/D={CL_pred[zi]/CD_pred[zi]:.1f}")
        print(f"  CL error:  {abs(CL_orig[zi]-CL_pred[zi]):.4f}")
        print(f"  CD error:  {abs(CD_orig[zi]-CD_pred[zi]):.6f}")
    else:
        print(f"  Original:  CL={CL_orig[zi]:.4f}  CD={CD_orig[zi]:.6f}")
        print(f"  Decoded:   FAILED at this alpha")
    print(f"{'='*50}")

    # Also check at design-ish angle
    di = np.argmin(np.abs(alphas - 2.0))
    print(f"\nCOMPARISON AT AoA = {alphas[di]:.1f} deg:")
    print(f"  Original:  CL={CL_orig[di]:.4f}  CD={CD_orig[di]:.6f}  L/D={CL_orig[di]/CD_orig[di]:.1f}")
    if np.isfinite(CL_pred[di]):
        print(f"  Decoded:   CL={CL_pred[di]:.4f}  CD={CD_pred[di]:.6f}  L/D={CL_pred[di]/CD_pred[di]:.1f}")
    else:
        print(f"  Decoded:   FAILED at this alpha")


# ============================================================
# PART 3 — GEOMETRY PLOT (Original vs Predicted)
# ============================================================

# Coords already unpacked in diagnostic section above
# upper_le2te and lower_le2te are ready for plotting

fig, axes = plt.subplots(1, 1, figsize=(10, 5))
axes.plot(orig_data[:, 0], orig_data[:, 1], 'k-', linewidth=2,
          label=f"Original {target_name} ({orig_data.shape[0]} pts)")
axes.plot(upper_le2te[:, 0], upper_le2te[:, 1], 'r--o', markersize=2,
          label="Decoded upper (40 pts)")
axes.plot(lower_le2te[:, 0], lower_le2te[:, 1], 'b--o', markersize=2,
          label="Decoded lower (40 pts)")
axes.set_aspect('equal')
axes.grid(True, alpha=0.3)
axes.set_xlabel("x/c")
axes.set_ylabel("y/c")
axes.set_title(f"Decoder Fidelity: Original vs Decoded {target_name}")
axes.legend()
axes.set_xlim(-0.05, 1.05)


# ============================================================
# PART 4 — CL and CD PLOTS
# ============================================================

fig2, (ax_cl, ax_cd) = plt.subplots(1, 2, figsize=(12, 5))

ax_cl.plot(alphas, CL_orig, '-s', markersize=4, label="Original")
ax_cl.plot(alphas, CL_pred, '-o', markersize=4, label="Decoded 80 pts")
ax_cl.set_xlabel("Angle of Attack (deg)")
ax_cl.set_ylabel("CL")
ax_cl.grid(True, alpha=0.3)
ax_cl.legend()
ax_cl.set_title("CL Comparison")

ax_cd.plot(alphas, CD_orig, '-s', markersize=4, label="Original")
ax_cd.plot(alphas, CD_pred, '-o', markersize=4, label="Decoded 80 pts")
ax_cd.set_xlabel("Angle of Attack (deg)")
ax_cd.set_ylabel("CD")
ax_cd.grid(True, alpha=0.3)
ax_cd.legend()
ax_cd.set_title("CD Comparison")

fig2.tight_layout()

plt.show()