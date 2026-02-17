import os
import numpy as np
import matplotlib.pyplot as plt
import aerosandbox as asb
import neuralfoil as nf

# ============================================================
# LOAD AIRFOILS
# ============================================================

# Load from txt file (n0012)
txt_dat = "airfoils_txt/n0012.txt"
txt_data = np.loadtxt(txt_dat, skiprows=1)
airfoil_txt = asb.Airfoil(name="n0012_txt", coordinates=txt_data)

# Load from original dat file
dat_file = "data/coord_seligFmt/n0012.dat"
orig_data = np.loadtxt(dat_file, skiprows=1)
airfoil_orig = asb.Airfoil(name="n0012_orig", coordinates=orig_data)

# ============================================================
# COMPUTE AERODYNAMICS
# ============================================================

alphas = np.linspace(-10, 15, 30)

velocity = 5.0
chord = 5.0
rho = 1000
mu = 1e-3

Re_water = rho * velocity * chord / mu


def compute_cl_cd(airfoil):
    out = nf.get_aero_from_airfoil(
        airfoil,
        alpha=alphas,        # Vectorized input (faster)
        Re=Re_water,
        model_size="large"
    )

    CL = np.asarray(out["CL"]).squeeze()
    CD = np.asarray(out["CD"]).squeeze()

    return CL, CD


CL_orig, CD_orig = compute_cl_cd(airfoil_orig)
CL_txt, CD_txt = compute_cl_cd(airfoil_txt)

# ============================================================
# PRINT COMPARISON
# ============================================================

zi = np.argmin(np.abs(alphas))  # index closest to 0°

print("\nAoA = 0°:")
print(f"Original (.dat): CL={CL_orig[zi]:.4f}, CD={CD_orig[zi]:.6f}")
print(f"From txt:        CL={CL_txt[zi]:.4f}, CD={CD_txt[zi]:.6f}")

# ============================================================
# GEOMETRY PLOT
# ============================================================

plt.figure(figsize=(8, 6))
plt.plot(orig_data[:, 0], orig_data[:, 1], 'k-', linewidth=2, label="Original .dat")
plt.plot(txt_data[:, 0], txt_data[:, 1], 'r--o', markersize=3, label="From txt")

plt.axis('equal')
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Airfoil Geometry: Original vs Txt")
plt.legend()

# ============================================================
# CL PLOT
# ============================================================

plt.figure()
plt.plot(alphas, CL_orig, '-s', label="Original (.dat)")
plt.plot(alphas, CL_txt, '-o', label="From txt")
plt.xlabel("Angle of Attack (deg)")
plt.ylabel("CL")
plt.grid(True)
plt.legend()
plt.title("CL Comparison")

# ============================================================
# CD PLOT
# ============================================================

plt.figure()
plt.plot(alphas, CD_orig, '-s', label="Original (.dat)")
plt.plot(alphas, CD_txt, '-o', label="From txt")
plt.xlabel("Angle of Attack (deg)")
plt.ylabel("CD")
plt.grid(True)
plt.legend()
plt.title("CD Comparison")

plt.show()