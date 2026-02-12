import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import winsound

# ============================================================
# SETTINGS
# ============================================================
IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 80
N_PARAMS = 6

AUTOENCODER_WEIGHTS = f"autoencoder_{N_PARAMS}params.weights.h5"
ENCODER_WEIGHTS = f"encoder_{N_PARAMS}params.weights.h5"
ENCODE_DIR = "airfoils_png"
TXT_DIR = "airfoils_txt"
OUT_CSV = f"airfoil_latent_params_{N_PARAMS}.csv"
OUT_NPY = f"airfoil_latent_params_{N_PARAMS}.npy"

np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# LOAD DATA
# ============================================================
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

def load_y_values(file_list, txt_dir, n_points=80):
    y_data = []
    for fname in file_list:
        fname_txt = os.path.splitext(fname)[0] + ".txt"
        txt_path = os.path.join(txt_dir, fname_txt)
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Missing {txt_path}")
        data = np.loadtxt(txt_path, skiprows=1)[:, 1]
        if len(data) != n_points:
            raise ValueError(f"{fname}: expected {n_points} points, got {len(data)}")
        y_data.append(data)
    return np.array(y_data, dtype=np.float32)

print("Loading PNGs...")
X_imgs, file_list = load_png_folder(ENCODE_DIR)
print(f"Loaded PNGs. Shape: {X_imgs.shape}")

print("Loading y-values...")
y_values = load_y_values(file_list, TXT_DIR, n_points=80)
print(f"Loaded y-values. Shape: {y_values.shape}")

# ============================================================
# BUILD FULL AUTOENCODER: PNG → 6 → 80
# ============================================================
print("\nBuilding full autoencoder (encoder + decoder)...")
inp_img = Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")

# --- ENCODER
x = Conv2D(32, (3,3), activation='relu', padding='valid')(inp_img)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu', padding='valid')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128, (3,3), activation='relu', padding='valid')(x)
x = MaxPooling2D((3,3))(x)
x = Conv2D(256, (3,3), activation='relu', padding='valid')(x)
x = MaxPooling2D((3,3))(x)
x = Flatten()(x)
x = Dense(1000, activation='relu')(x)
x = Dense(100, activation='relu')(x)
x = Dropout(0.05)(x)

latent = Dense(N_PARAMS, activation='linear', name=f"latent_{N_PARAMS}")(x)

# --- DECODER
x_dec = Dense(100, activation='relu')(latent)
x_dec = Dense(1000, activation='relu')(x_dec)
y_out = Dense(80, activation='linear', name="y_reconstructed")(x_dec)

# Full autoencoder
autoencoder = Model(inp_img, y_out)
autoencoder.compile(optimizer=Adam(5e-4), loss="mse")
autoencoder.summary()

# Extract encoder for later
encoder = Model(inp_img, latent)

# ============================================================
# TRAIN OR LOAD
# ============================================================
if os.path.exists(AUTOENCODER_WEIGHTS):
    autoencoder.load_weights(AUTOENCODER_WEIGHTS)
    print(f"\n✓ Loaded autoencoder from {AUTOENCODER_WEIGHTS}")
else:
    print("\n🔄 Training autoencoder (PNG → 6 params → 80 y-values)...")

    cbs = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]

    history = autoencoder.fit(
        X_imgs, y_values,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=cbs,
        verbose=1
    )

    autoencoder.save_weights(AUTOENCODER_WEIGHTS)
    print(f"\n✓ Saved autoencoder to {AUTOENCODER_WEIGHTS}")

    full_loss = autoencoder.evaluate(X_imgs, y_values, batch_size=BATCH_SIZE, verbose=0)
    print(f"✓ Final loss: {full_loss:.6f}")

    log_csv = "training_log.csv"
    best_val_loss = min(history.history["val_loss"])
    file_exists = os.path.exists(log_csv)

    with open(log_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["N_PARAMS", "final_loss", "best_val_loss"])
        writer.writerow([N_PARAMS, full_loss, best_val_loss])

# ============================================================
# EXTRACT ENCODER: PNG → 6 PARAMS
# ============================================================
print("\n📊 Encoding all airfoils with encoder...")
params_out = encoder.predict(X_imgs, batch_size=BATCH_SIZE, verbose=1)
print(f"✓ Encoded {len(params_out)} airfoils into {N_PARAMS}-D space")

# ============================================================
# SAVE OUTPUTS
# ============================================================
np.save(OUT_NPY, params_out)
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename"] + [f"p{i+1}" for i in range(N_PARAMS)])
    for fname, vec in zip(file_list, params_out):
        writer.writerow([fname] + vec.tolist())

print(f"✓ Saved to {OUT_CSV} and {OUT_NPY}")

encoder.save_weights(ENCODER_WEIGHTS)
print(f"✓ Saved encoder to {ENCODER_WEIGHTS}")

print("\n" + "="*50)
print("✓ ENCODER COMPLETE")
print("="*50)

winsound.MessageBeep()