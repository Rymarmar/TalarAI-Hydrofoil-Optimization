import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# ============================================================
# REPRODUCIBILITY SEEDS  [FIX: added to match encoder.py]
#
# WHY THIS MATTERS:
#   Without seeds, every retraining on a new device starts from
#   different random weights and converges to a different local
#   minimum. This changes the latent->y80 mapping, which changes
#   which foil NeuralFoil scores highest, which changes the
#   baseline selected from the lookup table.
#
# WHY SEEDS ALONE ARE NOT ENOUGH:
#   TF's oneDNN backend (visible in terminal: "oneDNN custom ops")
#   uses different floating-point operation orderings on different
#   CPUs (Intel vs AMD vs Apple Silicon). Seeds control random
#   INITIALIZATION, not rounding. Two CPUs with seed=42 will get
#   the same starting weights but slightly different gradients,
#   and may converge to slightly different solutions.
#
# CORRECT FIX: Train once, then COPY these files to all devices:
#   - autoencoder_6params.weights.h5
#   - encoder_6params.weights.h5
#   - decoder_model_6x100x1000x80.weights.h5  (see save fix below)
#   - airfoil_latent_params_6.csv
#   - outputs/best_baseline_foil_*.json
# ============================================================
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# -----------------------------
# Step 1: Load dataset
# -----------------------------
df = pd.read_csv("data/airfoil_latent_params_6.csv")  # columns: filename, p1..p6
params6 = df[['p1','p2','p3','p4','p5','p6']].values.astype(np.float32)

scaling_factor = 1

# Load all 80-point airfoils (y-values only)
y80_list = []
for fname in df['filename']:
    fname_txt = os.path.splitext(fname)[0] + ".txt"
    data = np.loadtxt(f"airfoils_txt/{fname_txt}", skiprows=1)[:, 1]  # y column
    y80_list.append(data)
y80 = np.array(y80_list, dtype=np.float32)

print("Dataset shapes:", params6.shape, y80.shape)

# -----------------------------
# Step 2: Train or load model (no scaler)
# -----------------------------
model_path = "decoder_model_6x100x1000x80.weights.h5"

if not os.path.exists(model_path):
    print("\n🚀 Training new decoder model...")

    # Architecture: 6 → 100 → 1000 → 80
    inp = Input(shape=(6,), name="params6")
    x = Dense(100, activation="relu")(inp)
    x = Dense(1000, activation="relu")(x)
    out = Dense(80, activation="linear")(x)
    decoder = Model(inp, out, name="testdecoder_6x100x1000x80")

    decoder.compile(optimizer=Adam(1e-3), loss="mse")

    history = decoder.fit(
        params6, y80,
        epochs=500,
        batch_size=16,
        validation_split=0.1,
        verbose=1
    )

    # [FIX] Use save_weights() instead of save() to match the .weights.h5
    # filename. decoder.save() saves the full model (architecture + weights)
    # in Keras format, but the filename ends in .weights.h5 which implies
    # weights-only format. On different TF versions, save() behaves
    # differently with this extension, causing load_model() to fail or
    # produce corrupted state on other devices.
    # save_weights() + load_weights() is consistent across TF versions.
    decoder.save_weights(model_path)
else:
    print("\n Found trained model. Loading it instead...")
    # Rebuild the same architecture, then load the saved weights.
    # This is required when using save_weights() instead of save().
    inp = Input(shape=(6,), name="params6")
    x = Dense(100, activation="relu")(inp)
    x = Dense(1000, activation="relu")(x)
    out = Dense(80, activation="linear")(x)
    decoder = Model(inp, out, name="testdecoder_6x100x1000x80")
    decoder.load_weights(model_path)

# -----------------------------
# Step 3: Predict and plot every 500th airfoil
# -----------------------------
x_pred = np.linspace(1, 0, 40)  # Flipped: 1 to 0
indices = np.arange(0, len(params6), 500)  # every 500th airfoil

for i in indices:
    y_pred = decoder.predict(params6[i:i+1])[0]

    y_upper, y_lower = y_pred[:40], y_pred[40:]
    y_upper_scaled = y_upper * scaling_factor
    y_lower_scaled_rev = (y_lower * scaling_factor)[::-1]

    y_orig = y80[i]
    y_upper_orig, y_lower_orig = y_orig[:40], y_orig[40:]
    y_lower_orig_rev = y_lower_orig[::-1]

    plt.figure(figsize=(8, 8))
    plt.plot(x_pred, y_upper_orig, 'b-', label='Original Upper', linewidth=2)
    plt.plot(x_pred, y_lower_orig_rev, 'b-', label='Original Lower', linewidth=2)
    plt.plot(x_pred, y_upper_scaled, 'r--', label='Pred Upper', linewidth=2)
    plt.plot(x_pred, y_lower_scaled_rev, 'r--', label='Pred Lower', linewidth=2)
    plt.axis('off')
    plt.gca().set_aspect('equal')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.3, 0.3)
    plt.legend()
    plt.title(f"Airfoil #{i} ({df['filename'][i]})")
    plt.tight_layout()

    print(f"\nPredicted y-values for Airfoil {i}:")
    print(y_pred)

plt.show()
decoder.summary()