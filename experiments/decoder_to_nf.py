import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

import aerosandbox as asb
import neuralfoil as nf


# ============================================================
# PART 1 — LOAD LATENT PARAMS + DECODE AIRFOIL
# ============================================================
df = pd.read_csv("airfoil_latent_params_6.csv")
params6 = df[['p1','p2','p3','p4','p5','p6']].values.astype(np.float32)

scaler_path = "testscaler_params.npz"
model_path = "testdecoder_model_6x100x1000x80.h5"

decoder = load_model(model_path)
scaler_data = np.load(scaler_path)
scaler = StandardScaler()
scaler.mean_ = scaler_data["mean"]
scaler.scale_ = scaler_data["scale"]

# 🔎 find naca2410
target_name = "naca2410"
matches = df[df["filename"].str.contains(target_name, case=False)]
if len(matches) == 0:
    raise ValueError("No naca2410 in dataset!")
i = matches.index[0]

# 🔄 decode 80 y-points
y_pred = decoder.predict(scaler.transform(params6[i:i+1]))[0]

y_upper = y_pred[:40]
y_lower = y_pred[40:][::-1]

# X-coordinates (reversed to flip leading/trailing edges)
x_upper = np.linspace(1, 0, 40)
x_lower = np.linspace(0, 1, 40)

x_combined = np.concatenate([x_upper, x_lower])
y_combined = np.concatenate([y_upper, y_lower[::-1]])

# Save predicted airfoil
output_csv = "naca2410_predicted_airfoil.csv"
with open(output_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['x','y'])
    for xi, yi in zip(x_combined, y_combined):
        w.writerow([xi, yi])

print("Saved predicted foil:", output_csv)

# Load original airfoil
orig_dat = "coord_seligFmt/naca2410.dat"
orig_data = np.loadtxt(orig_dat, skiprows=1)

# ============================================================
# PART 2 — AEROSANDBOX + NEURALFOIL SETUP
# ============================================================

# --- Original
airfoil_orig = asb.Airfoil(name="naca2410_orig", coordinates=orig_data)

# --- Predicted
pred_data = np.column_stack([x_combined, y_combined])
airfoil_pred = asb.Airfoil(name="naca2410_pred", coordinates=pred_data)


# ============================================================
# FLOW CONDITIONS
# ============================================================
alphas = np.linspace(-10, 15, 30)
velocity = 5.0
chord = 5.0
rho = 1000
mu = 1e-3
Re_water = rho * velocity * chord / mu

def compute_cl_cd(airfoil):
    CLs, CDs = [], []
    for alpha in alphas:
        out = nf.get_aero_from_airfoil(
            airfoil,
            alpha=alpha,
            Re=Re_water,
            model_size="large"
        )
        CLs.append(float(out["CL"]))
        CDs.append(float(out["CD"]))
    return np.array(CLs), np.array(CDs)

# compute aero
CL_orig, CD_orig = compute_cl_cd(airfoil_orig)
CL_pred, CD_pred = compute_cl_cd(airfoil_pred)

# print zero AoA comparison
zi = np.argmin(np.abs(alphas))
print("\nAoA = 0°:")
print(f"Original:  CL={CL_orig[zi]:.4f}, CD={CD_orig[zi]:.6f}")
print(f"Predicted: CL={CL_pred[zi]:.4f}, CD={CD_pred[zi]:.6f}")


# ============================================================
# PART 3 — GEOMETRY PLOT (Original vs Predicted)
# ============================================================
plt.figure(figsize=(8,6))
plt.plot(orig_data[:,0], orig_data[:,1], 'k-', linewidth=2, label="Original naca2410")
plt.plot(pred_data[:,0], pred_data[:,1], 'r--o', markersize=3, label="Predicted 80 pts")

plt.axis('equal')
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Airfoil Geometry: Original vs Predicted (80-point decoded)")
plt.legend()


# ============================================================
# PART 4 — CL and CD PLOTS
# ============================================================
plt.figure()
plt.plot(alphas, CL_orig, '-s', label="Original")
plt.plot(alphas, CL_pred, '-o', label="Predicted 80 pts")
plt.xlabel("Angle of Attack (deg)")
plt.ylabel("CL")
plt.grid(True)
plt.legend()
plt.title("CL Comparison")

plt.figure()
plt.plot(alphas, CD_orig, '-s', label="Original")
plt.plot(alphas, CD_pred, '-o', label="Predicted 80 pts")
plt.xlabel("Angle of Attack (deg)")
plt.ylabel("CD")
plt.grid(True)
plt.legend()
plt.title("CD Comparison")

plt.show()