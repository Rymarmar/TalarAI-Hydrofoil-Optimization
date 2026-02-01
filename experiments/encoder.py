import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# -----------------------------
# SETTINGS
# -----------------------------
IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 50
X_SAMPLES = np.linspace(0.0, 1.0, 40).astype(np.float32)  # 40 x-values
WEIGHTS_PATH = "encoder.weights.h5"
ENCODE_DIR = "airfoils_png"  # folder with PNGs
TXT_DIR = "airfoils_txt"     # folder with 80-point y-values
OUT_CSV = "airfoil_latent_params.csv"
OUT_NPY = "airfoil_latent_params.npy"

np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------
# LOAD PNG FOLDER
# -----------------------------
def load_png_folder(folder, img_size=IMG_SIZE):
    files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    files.sort()
    if not files:
        raise RuntimeError(f"No PNGs found in {folder}")
    X = []
    for f in files:
        p = os.path.join(folder, f)
        img = load_img(p, target_size=(img_size, img_size), color_mode="grayscale")
        arr = img_to_array(img) / 255.0
        X.append(arr)
    return np.array(X, dtype="float32"), files

X_imgs, file_list = load_png_folder(ENCODE_DIR)
print("Loaded PNGs. Shape:", X_imgs.shape)

# -----------------------------
# ANALYTIC DECODER
# -----------------------------
def decoder_from_conics(args):
    params6, x_grid = args
    A = params6[:, 0:1]; B = params6[:, 1:2]; C = params6[:, 2:3]
    D = params6[:, 3:4]; E = params6[:, 4:5]; F = params6[:, 5:6]

    Xg = tf.expand_dims(x_grid, axis=0)
    Xg = tf.tile(Xg, [tf.shape(params6)[0], 1])

    b_lin = B * Xg + E
    c_const = A * (Xg**2) + D * Xg + F
    disc = tf.maximum(0.0, b_lin**2 - 4.0 * C * c_const)
    sqrt_disc = tf.sqrt(disc + 1e-12)
    denom = 2.0 * C + 1e-12

    y_top = (-b_lin + sqrt_disc) / denom
    y_bot = (-b_lin - sqrt_disc) / denom
    return tf.concat([y_top, y_bot], axis=1)

# -----------------------------
# BUILD CNN ENCODER + DECODER
# -----------------------------
inp = Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")

x = Conv2D(32, (3,3), activation='relu', padding='valid')(inp)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu', padding='valid')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128, (3,3), activation='relu', padding='valid')(x)
x = MaxPooling2D((3,3))(x)
x = Conv2D(256, (3,3), activation='relu', padding='valid')(x)
x = MaxPooling2D((3,3))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.05)(x)

params_pred = Dense(6, activation='linear', name="params6")(x)
x_grid_const = tf.constant(X_SAMPLES, dtype=tf.float32)
y80_pred = Lambda(decoder_from_conics)([params_pred, x_grid_const])

model = Model(inp, outputs={"y80": y80_pred, "p6": params_pred})
model.compile(optimizer=Adam(5e-4),
              loss={"y80":"mse","p6":"mse"},
              loss_weights={"y80":1.0,"p6":0.1})
model.summary()

# -----------------------------
# TRAIN IF WEIGHTS DO NOT EXIST
# -----------------------------
if os.path.exists(WEIGHTS_PATH):
    # Simpler load for Keras 3 – no by_name / skip_mismatch
    model.load_weights(WEIGHTS_PATH)
    print("Loaded encoder weights from", WEIGHTS_PATH)
else:
    print("No encoder weights found. Training encoder from scratch...")

    # Load corresponding 80-point airfoils
    y80_list = []
    for fname in file_list:
        fname_txt = os.path.splitext(fname)[0] + ".txt"
        data = np.loadtxt(os.path.join(TXT_DIR, fname_txt), skiprows=1)[:,1]
        y80_list.append(data)
    y80 = np.array(y80_list, dtype=np.float32)
    params = np.zeros((len(file_list),6), dtype=np.float32)  # placeholder for multi-target

    cbs = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_imgs,
        {"y80": y80, "p6": params},
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=cbs,
        verbose=1
    )

    # Sanity-check plots for first 3 airfoils
    y_pred = model.predict(X_imgs[:5], verbose=0)["y80"]
    N_X = 40
    for i in range(3):
        yp = y_pred[i]; yt = y80[i]
        plt.figure(figsize=(5,4))
        plt.plot(X_SAMPLES, yt[:N_X], label="top true")
        plt.plot(X_SAMPLES, yt[N_X:], label="bot true")
        plt.plot(X_SAMPLES, yp[:N_X], '--', label="top pred")
        plt.plot(X_SAMPLES, yp[N_X:], '--', label="bot pred")
        plt.title(f"Sample {i}: pred vs true"); plt.legend()
        plt.xlabel("x"); plt.ylabel("y"); plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"compare_{i}.png", dpi=150); plt.close()

    model.save_weights(WEIGHTS_PATH)
    print("Saved encoder weights to", WEIGHTS_PATH)

# -----------------------------
# ENCODE PNGs → 6 PARAMETERS
# -----------------------------
encoder = Model(inp, params_pred)
params6 = encoder.predict(X_imgs, batch_size=BATCH_SIZE, verbose=1)
print("Encoded params shape:", params6.shape)

# Save CSV and NPY
np.save(OUT_NPY, params6)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filename","p1","p2","p3","p4","p5","p6"])
    for fname, vec in zip(file_list, params6):
        w.writerow([fname]+list(map(float, vec)))

print(f"Saved: {OUT_CSV} and {OUT_NPY}")
