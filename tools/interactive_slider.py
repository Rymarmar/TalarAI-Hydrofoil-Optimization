import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# -----------------------------
# Load decoder and scaler
# -----------------------------
scaler_path = "scaler_params.npz"
model_path = "decoder_model_6x100x1000x80.h5"

decoder = load_model(model_path, compile=False)
scaler_data = np.load(scaler_path)
scaler = StandardScaler()
scaler.mean_ = scaler_data["mean"]
scaler.scale_ = scaler_data["scale"]

x_pred = np.linspace(0, 1, 40)
scaling_factor = 1

# -----------------------------
# Initial parameters
# -----------------------------
init_params = [0.13320291, -0.46922102, 0.04649611,
               -0.81211898, 0.15189299, 0.8585924]
# -----------------------------
# Create figure
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(left=0.1, bottom=0.35)
line_upper, = ax.plot([], [], 'r--', label='Pred Upper', linewidth=2)
line_lower, = ax.plot([], [], 'r--', label='Pred Lower', linewidth=2)
ax.axis('off')
ax.set_aspect('equal')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.3, 0.3)
ax.legend()
plt.title("Interactive Hydrofoil")

# -----------------------------
# Slider axes
# -----------------------------
axcolor = 'lightgoldenrodyellow'
slider_axes = []
sliders = []
param_names = ['p1','p2','p3','p4','p5','p6']
param_ranges = [(-1.0, 1.0)] * 6

for i, (name, (vmin, vmax)) in enumerate(zip(param_names, param_ranges)):
    ax_slider = plt.axes([0.1, 0.25 - i*0.03, 0.8, 0.02], facecolor=axcolor)
    slider = Slider(ax_slider, name, vmin, vmax, valinit=init_params[i], valstep=0.0001)
    sliders.append(slider)

# -----------------------------
# Update function
# -----------------------------
def update(val):
    params = np.array([[s.val for s in sliders]])
    params_scaled = scaler.transform(params)
    y_pred = decoder.predict(params_scaled)[0]

    y_upper = y_pred[:40] * scaling_factor
    y_lower = y_pred[40:] * scaling_factor
    y_lower_rev = y_lower[::-1]

    line_upper.set_data(x_pred, y_upper)
    line_lower.set_data(x_pred, y_lower_rev)
    fig.canvas.draw_idle()

# -----------------------------
# Connect sliders
# -----------------------------
for s in sliders:
    s.on_changed(update)

# Initial draw
update(None)
plt.show()
