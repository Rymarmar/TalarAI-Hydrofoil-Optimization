import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Paths (match YOUR repo)
# ============================================================
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dat_folder = os.path.join(REPO_ROOT, "data", "coord_seligFmt")   # <-- put .dat files here
outdir = os.path.join(REPO_ROOT, "airfoils_png")
txt_outdir = os.path.join(REPO_ROOT, "airfoils_txt")

os.makedirs(outdir, exist_ok=True)
os.makedirs(txt_outdir, exist_ok=True)

# ============================================================
# Settings
# ============================================================
n_points = 40
dpi = 100
img_size = (2.56, 2.56)  # 2.56 inches * 100 dpi = 256 pixels

dat_files = sorted([
    os.path.join(dat_folder, f)
    for f in os.listdir(dat_folder)
    if f.endswith(".dat")
])

print(f"Found {len(dat_files)} airfoil files in {dat_folder}")

count = 0
for file_path in dat_files:
    try:
        with open(file_path, "r", errors="ignore") as f:
            lines = f.readlines()

        coords = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    coords.append([float(parts[0]), float(parts[1])])
                except:
                    pass

        coords = np.array(coords)
        if len(coords) < 10:
            continue

        # leading edge index (min x)
        le_idx = np.argmin(coords[:, 0])
        upper = coords[:le_idx + 1]
        lower = coords[le_idx:]

        # sort by x
        upper = upper[np.argsort(upper[:, 0])]
        lower = lower[np.argsort(lower[:, 0])]

        # interpolate to standard x grid
        x_common = np.linspace(0, 1, n_points)
        yu = np.interp(x_common, upper[:, 0], upper[:, 1])
        yl = np.interp(x_common, lower[:, 0], lower[:, 1])

        # IMPORTANT: y ordering used for TXT (must match encoder/decoder expectations)
        # Here we store 80 points as:
        #   x = [x_common reversed (1->0), x_common (0->1)]
        #   y = [yl reversed (lower), yu (upper)]
        x = np.concatenate([x_common[::-1], x_common])
        y = np.concatenate([yl[::-1], yu])

        # normalize x to [0,1]
        x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-12)

        # --- Plot PNG (just black lines on white background)
        fig, ax = plt.subplots(figsize=img_size, dpi=dpi)
        ax.plot(x_common, yu, 'k', linewidth=2)
        ax.plot(x_common, yl, 'k', linewidth=2)
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.3, 0.3)

        name = os.path.splitext(os.path.basename(file_path))[0]

        # save PNG
        save_path = os.path.join(outdir, f"{name}.png")
        fig.savefig(save_path, dpi=dpi, bbox_inches=None, pad_inches=0, facecolor='white')
        plt.close(fig)

        # save TXT
        txt_path = os.path.join(txt_outdir, f"{name}.txt")
        np.savetxt(txt_path, np.column_stack([x, y]), fmt="%.6f", header="x y", comments='')

        count += 1
        if count % 50 == 0:
            print(f"Saved {count} airfoils...")

    except Exception as e:
        print(f"Skipping {file_path}: {e}")

print(f"✅ Done! Saved {count} PNGs to {outdir} and TXT files to {txt_outdir}")
