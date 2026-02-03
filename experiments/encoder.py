import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ============================================================
# Paths (match YOUR repo)
# ============================================================
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ENCODE_DIR = os.path.join(REPO_ROOT, "airfoils_png")
TXT_DIR = os.path.join(REPO_ROOT, "airfoils_txt")

DATA_DIR = os.path.join(REPO_ROOT, "data")
PIPELINE_DIR = os.path.join(REPO_ROOT, "pipeline")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PIPELINE_DIR, exist_ok=True)

# Outputs (match final pipeline expectations)
OUT_CSV = os.path.join(DATA_DIR, "airfoil_latent_params.csv")
OUT_NPY = os.path.join(DATA_DIR, "airfoil_latent_params.npy")

# IMPORTANT: Keras requires save_weights filenames to end with ".weights.h5"
AUTOENCODER_WEIGHTS = os.path.join(PIPELINE_DIR, "autoencoder_6params.weights.h5")
ENCODER_WEIGHTS = os.path.join(PIPELINE_DIR, "encoder_6params.weights.h5")

# Safety checkpoint (saves best weights during training)
CHECKPOINT_PATH = os.path.join(PIPELINE_DIR, "autoencoder_6params_best.weights.h5")

# ============================================================
# Settings
# ============================================================
IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 80
N_PARAMS = 6

np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# Load data
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
print(f"Loaded PNGs: {X_imgs.shape}")

print("Loading y-values...")
y_values = load_y_values(file_list, TXT_DIR, n_points=80)
print(f"Loaded y-values: {y_values.shape}")

# ============================================================
# Build autoencoder: PNG -> 6 -> 80
# ============================================================
inp_img = Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")

# Encoder
x = Conv2D(32, (3, 3), activation="relu", padding="valid")(inp_img)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation="relu", padding="valid")(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation="relu", padding="valid")(x)
x = MaxPooling2D((3, 3))(x)
x = Conv2D(256, (3, 3), activation="relu", padding="valid")(x)
x = MaxPooling2D((3, 3))(x)
x = Flatten()(x)
x = Dense(1000, activation="relu")(x)
x = Dense(100, activation="relu")(x)
x = Dropout(0.05)(x)

latent = Dense(N_PARAMS, activation="linear", name="latent")(x)

# Decoder (latent -> y80)
x_dec = Dense(100, activation="relu")(latent)
x_dec = Dense(1000, activation="relu")(x_dec)
y_out = Dense(80, activation="linear", name="y_reconstructed")(x_dec)

autoencoder = Model(inp_img, y_out)
autoencoder.compile(optimizer=Adam(5e-4), loss="mse")

encoder = Model(inp_img, latent)

# ============================================================
# Train or load
# ============================================================
if os.path.exists(AUTOENCODER_WEIGHTS):
    autoencoder.load_weights(AUTOENCODER_WEIGHTS)
    print(f"✓ Loaded autoencoder weights from {AUTOENCODER_WEIGHTS}")
else:
    print("🔄 Training autoencoder...")

    cbs = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        # Saves the best weights DURING training (so you never lose progress again)
        ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
    ]

    autoencoder.fit(
        X_imgs, y_values,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=cbs,
        verbose=1
    )

    # Save final weights safely
    autoencoder.save_weights(AUTOENCODER_WEIGHTS)
    print(f"✓ Saved final autoencoder weights to {AUTOENCODER_WEIGHTS}")

# ============================================================
# Encode all PNGs -> latent params
# ============================================================
print("📊 Encoding all airfoils...")
params_out = encoder.predict(X_imgs, batch_size=BATCH_SIZE, verbose=1)
print(f"✓ Encoded {len(params_out)} airfoils into {N_PARAMS}-D latent space")

# ============================================================
# Save outputs to data/
# ============================================================
np.save(OUT_NPY, params_out)
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename"] + [f"p{i+1}" for i in range(N_PARAMS)])
    for fname, vec in zip(file_list, params_out):
        writer.writerow([fname] + vec.tolist())

encoder.save_weights(ENCODER_WEIGHTS)
print(f"✓ Saved encoder weights to {ENCODER_WEIGHTS}")

print(f"✓ Saved latents to:\n  {OUT_CSV}\n  {OUT_NPY}")
print("✅ ENCODER COMPLETE")
