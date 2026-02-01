# Use a Convolutional Neural Network (CNN) to learn how to describe ellipses
# using 6 mathematical parameters, then reconstruct them back to check accuracy
#
# Encoder (image → parameters) + analytic decoder (parameters → 80 y-values).
# Matches the slide shapes: 32x127x127 → 64x62x62 → 128x20x20 → 256x6x6 → Flatten(9216)

import os
import csv                                # >>> ADDED: to write CSV of 6 params
import argparse                           # >>> ADDED: simple flags (skip training, set folder)
import numpy as np
import matplotlib
matplotlib.use("Agg")    # allows saving plots without needing a display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ------------------------------------------------------------------------------
# STEP 0: Set up folders, random seeds, constants, and simple CLI flags
# ------------------------------------------------------------------------------

ap = argparse.ArgumentParser()
# --no-train lets you skip ellipse training and only encode your PNG folder
ap.add_argument("--no-train", action="store_true",
                help="Skip ellipse training; just load weights and encode a PNG folder.")
# Path to your airfoil PNG folder (Windows path OK)
ap.add_argument("--encode_dir", type=str,
                default=r"C:\Users\benma\Documents\NeuralFoil\TalarAI\airfoils_png",
                help="Folder of pre-interpolated airfoil PNGs to encode (output 6 params).")
ap.add_argument("--out_csv", type=str, default="airfoil_latent_params.csv")
ap.add_argument("--out_npy", type=str, default="airfoil_latent_params.npy")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=32)
args = ap.parse_args()

np.random.seed(42)
tf.random.set_seed(42)

IMG_SIZE = 256          # image resolution for each ellipse PNG (must be 256 to match slide dims)
DPI = 100               # DPI for Matplotlib figures
OUTDIR = "ellipse_images"
os.makedirs(OUTDIR, exist_ok=True)

N_IMAGES = 1000         # how many ellipses to create (dataset size)
N_X = 40                # number of x-values between 0 and 1
X_SAMPLES = np.linspace(0.0, 1.0, N_X).astype("float32")  # 40 evenly spaced x-values

# ------------------------------------------------------------------------------
# STEP 1: Generate one ellipse (image + labels)  [unchanged, used for training]
# ------------------------------------------------------------------------------

def make_one_ellipse(i, a, b):
    inside = np.maximum(0.0, 1.0 - (X_SAMPLES / a) ** 2)
    y_top =  b * np.sqrt(inside)
    y_bot = -b * np.sqrt(inside)
    y80 = np.concatenate([y_top, y_bot]).astype("float32")  # 40 top + 40 bottom = 80 total

    # Conic parameters for centered ellipse: A=1/a^2, C=1/b^2, B=D=E=0, F=-1
    params6 = np.array([1.0/(a*a), 0.0, 1.0/(b*b), 0.0, 0.0, -1.0], dtype="float32")

    t = np.linspace(0, 2*np.pi, 800)
    x_curve = a * np.cos(t)
    y_curve = b * np.sin(t)

    fig, ax = plt.subplots(figsize=(IMG_SIZE/DPI, IMG_SIZE/DPI), dpi=DPI)
    ax.plot(x_curve, y_curve, 'k', linewidth=2)
    ax.set_aspect('equal'); ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.axis('off')
    save_path = os.path.join(OUTDIR, f"ellipse_{i:04d}.png")
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close(fig)

    return save_path, params6, y80

# ------------------------------------------------------------------------------
# STEP 2: Create or load the ellipse dataset  [unchanged]
# ------------------------------------------------------------------------------

npz_path = "ellipse_data.npz"

if not os.path.exists(npz_path):
    img_paths, params_list, y_list = [], [], []
    for i in range(N_IMAGES):
        a = np.random.uniform(0.35, 0.95)
        b = np.random.uniform(0.25, 0.95)
        pth, p6, y80 = make_one_ellipse(i, a, b)
        img_paths.append(pth); params_list.append(p6); y_list.append(y80)

    params = np.stack(params_list, axis=0)  # (1000, 6)
    y80 = np.stack(y_list, axis=0)          # (1000, 80)

    imgs = []
    for p in img_paths:
        im = load_img(p, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
        arr = img_to_array(im) / 255.0
        imgs.append(arr)
    X_imgs = np.array(imgs, dtype="float32")  # (1000, 256, 256, 1)
    np.savez_compressed(npz_path, X_imgs=X_imgs, params=params, y80=y80, X_SAMPLES=X_SAMPLES)
else:
    data = np.load(npz_path)
    X_imgs = data["X_imgs"]; params = data["params"]; y80 = data["y80"]

print("Ellipse dataset shapes:", X_imgs.shape, params.shape, y80.shape)

# ------------------------------------------------------------------------------
# STEP 3: Define the CNN model (Encoder + Decoder)
#   NOTE: We keep references to the encoder's input and the Dense(6) layer so
#         we can reuse them later to build an encoder-only model for encoding PNGs.
# ------------------------------------------------------------------------------

def decoder_from_conics(args):
    """
    Analytic decoder: reconstruct 80 y-values from predicted [A,B,C,D,E,F].
    Solve for y in: C*y^2 + (B*x + E)*y + (A*x^2 + D*x + F) = 0
      y = [-(Bx+E) ± sqrt((Bx+E)^2 - 4C(Ax^2 + Dx + F))] / (2C)
    """
    params6, x_grid = args
    A = params6[:, 0:1]; B = params6[:, 1:2]; C = params6[:, 2:3]
    D = params6[:, 3:4]; E = params6[:, 4:5]; F = params6[:, 5:6]

    Xg = tf.expand_dims(x_grid, axis=0)
    Xg = tf.tile(Xg, [tf.shape(params6)[0], 1])   # (B, 40)

    b_lin = B * Xg + E
    c_const = A * (Xg**2) + D * Xg + F
    disc = tf.maximum(0.0, b_lin**2 - 4.0 * C * c_const)
    sqrt_disc = tf.sqrt(disc + 1e-12)
    denom = 2.0 * C + 1e-12

    y_top = (-b_lin + sqrt_disc) / denom
    y_bot = (-b_lin - sqrt_disc) / denom
    return tf.concat([y_top, y_bot], axis=1)    # (B, 80)

# --- CNN Encoder (matches slide shapes exactly) ---
inp = Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")  # keep a handle to 'inp'

# 1) 256 -> 254 -> 127   (32×127×127)
x = Conv2D(32, (3,3), activation='relu', padding='valid', name="conv1")(inp)
x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool1")(x)

# 2) 127 -> 125 -> 62    (64×62×62)
x = Conv2D(64, (3,3), activation='relu', padding='valid', name="conv2")(x)
x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool2")(x)

# 3) 62 -> 60 -> 20      (128×20×20)
x = Conv2D(128, (3,3), activation='relu', padding='valid', name="conv3")(x)
x = MaxPooling2D(pool_size=(3,3), strides=(3,3), name="pool3")(x)

# 4) 20 -> 18 -> 6       (256×6×6)
x = Conv2D(256, (3,3), activation='relu', padding='valid', name="conv4")(x)
x = MaxPooling2D(pool_size=(3,3), strides=(3,3), name="pool4")(x)

# Flatten 256×6×6 -> 9216 (matches slide)
x = Flatten(name="flatten")(x)
x = Dense(128, activation='relu', name="fc1")(x)
x = Dropout(0.05, name="drop")(x)        # >>> ADDED: light regularization (no harm)

# --- Multi-task head: predict 6 params AND 80 y-values (decoder) ---
params_pred = Dense(6, activation='linear', name="params6")(x)   # keep handle to this layer

# Decoder uses the analytic formula to output 80 y-values
x_grid_const = tf.constant(X_SAMPLES, dtype=tf.float32)
y80_pred = Lambda(decoder_from_conics, name="decoder")([params_pred, x_grid_const])

# Two-output model for ellipse training
model = Model(inp, outputs={"y80": y80_pred, "p6": params_pred}, name="ellipse_encoder_decoder")

model.compile(
    optimizer=Adam(5e-4),
    loss={"y80": "mse", "p6": "mse"},
    loss_weights={"y80": 1.0, "p6": 0.1},
)
model.summary()

# ------------------------------------------------------------------------------
# STEP 4: Train the CNN (with callbacks)  [or skip with --no-train]
#   After training, we save **encoder-only** weights so we can encode any PNG folder.
# ------------------------------------------------------------------------------

if not args.no_train:
    cbs = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]
    history = model.fit(
        X_imgs,
        {"y80": y80, "p6": params},     # multi-target training
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch,
        shuffle=True,
        callbacks=cbs,
        verbose=1
    )

    # Little sanity-check plots (unchanged)
    pred = model.predict(X_imgs[:5], verbose=0)
    y_pred = pred["y80"]
    for i in range(3):
        yp = y_pred[i]; yt = y80[i]
        y_top_p, y_bot_p = yp[:N_X], yp[N_X:]
        y_top_t, y_bot_t = yt[:N_X], yt[N_X:]
        plt.figure(figsize=(5,4))
        plt.plot(X_SAMPLES, y_top_t, label="top true")
        plt.plot(X_SAMPLES, y_bot_t, label="bot true")
        plt.plot(X_SAMPLES, y_top_p, '--', label="top pred")
        plt.plot(X_SAMPLES, y_bot_p, '--', label="bot pred")
        plt.title(f"Sample {i}: pred vs true"); plt.legend()
        plt.xlabel("x in [0,1]"); plt.ylabel("y"); plt.grid(True); plt.tight_layout()
        plt.savefig(f"compare_{i}.png", dpi=150); plt.close()

    # >>> ADDED: save encoder-only weights so we can reuse the encoder to encode folders
    model.save_weights("encoder.weights.h5")   # <- must end with .weights.h5
    print("Saved encoder weights to encoder.weights.h5")


# ------------------------------------------------------------------------------
# STEP 5: ENCODE a folder of airfoil PNGs → output 6 parameters (CSV + NPY)
#   This is the NEW part: use the encoder to turn each PNG into 6 numbers.
#   If you skipped training, we load encoder_weights.h5; otherwise we reuse the ones just trained.
# ------------------------------------------------------------------------------

def build_encoder_only():
    """Create an encoder-only model that outputs the 6-D vector from 'params6'."""
    # We already have 'inp' and the 'params6' layer tensor (params_pred) above.
    return Model(inp, params_pred, name="airfoil_encoder")

def load_png_folder(folder, img_size=256):
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

# Build encoder-only graph
encoder = build_encoder_only()

# If we skipped training, load weights
if args.no_train:
    if not os.path.exists("encoder.weights.h5"):
        raise FileNotFoundError(
            "encoder.weights.h5 not found. Train once (without --no-train) to create it."
        )
    encoder.load_weights("encoder.weights.h5")
    print("Loaded encoder weights from encoder.weights.h5")

# Load your airfoil PNGs
print(f"Encoding PNG folder: {args.encode_dir}")
X_encode, file_list = load_png_folder(args.encode_dir, img_size=IMG_SIZE)
print("Found", len(file_list), "PNGs. Shape:", X_encode.shape)

# Run encoder → 6 numbers per image
params6 = encoder.predict(X_encode, batch_size=args.batch, verbose=1)
print("Encoded params shape:", params6.shape)  # (N, 6)

# Save NPY
np.save(args.out_npy, params6)
# Save CSV with filenames
with open(args.out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filename", "p1", "p2", "p3", "p4", "p5", "p6"])
    for fname, vec in zip(file_list, params6):
        w.writerow([fname] + list(map(float, vec)))

print(f"Saved: {args.out_csv} and {args.out_npy}")
print("Done.")