"""
build_lookup_table.py

Builds a lookup table using NeuralFoil
directly from Selig .dat files.

Outputs:
    outputs/lookup_table.csv
    outputs/top_200_best_foils.csv
"""

import os
import numpy as np
import pandas as pd
import neuralfoil as nf


# ------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------

SELIG_FOLDER = "data/coord_seligFmt"
OUTPUT_FOLDER = "outputs"
LOOKUP_FILE = os.path.join(OUTPUT_FOLDER, "lookup_table.csv")
TOP200_FILE = os.path.join(OUTPUT_FOLDER, "top_200_best_foils.csv")
LATENT_FILE = os.path.join(OUTPUT_FOLDER, "all_latent_params_numeric.csv")

RE_LIST = [50000, 1e5, 2e5, 5e5, 1e6]
ALPHAS = np.arange(-2, 11, 2)


# ------------------------------------------------------------
# LOOKUP TABLE BUILDER
# ------------------------------------------------------------

def build_lookup_table():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    airfoil_files = [
        f for f in os.listdir(SELIG_FOLDER)
        if f.endswith(".dat")
    ]

    if not airfoil_files:
        raise ValueError("No .dat files found in folder.")

    results = []

    print(f"Found {len(airfoil_files)} airfoils.")
    print("Evaluating with NeuralFoil (dat file mode)...\n")

    for foil_file in airfoil_files:

        foil_name = foil_file.replace(".dat", "")
        foil_path = os.path.join(SELIG_FOLDER, foil_file)

        print(f"Evaluating {foil_name}...")

        for Re_water in RE_LIST:

            try:
                aero = nf.get_aero_from_dat_file(
                    foil_path,
                    alpha=ALPHAS,
                    Re=Re_water,
                    model_size="xlarge"
                )

                CL_array = np.asarray(aero["CL"]).flatten()
                CD_array = np.asarray(aero["CD"]).flatten()

                for i, alpha in enumerate(ALPHAS):

                    CL = float(CL_array[i])
                    CD = float(CD_array[i])

                    if not np.isfinite(CL) or not np.isfinite(CD) or CD <= 0:
                        L_over_D = np.nan
                        CD_over_CL = np.nan
                    else:
                        L_over_D = CL / CD
                        CD_over_CL = CD / CL if CL != 0 else np.nan

                    results.append({
                        "foil": foil_name,
                        "Re": Re_water,
                        "alpha": alpha,
                        "CL": CL,
                        "CD": CD,
                        "L_over_D": L_over_D,
                        "CD_over_CL": CD_over_CL
                    })

            except Exception as e:
                print(f"  Failed at Re={Re_water}: {e}")
                continue

    if not results:
        print("⚠️ No aerodynamic results produced.")
        return

    # ------------------------------------------------------------
    # SAVE LOOKUP TABLE
    # ------------------------------------------------------------

    df = pd.DataFrame(results)

    print(f"\nSaving lookup table to '{LOOKUP_FILE}'...")
    df.to_csv(LOOKUP_FILE, index=False)

    print("Lookup table complete.")


    # ------------------------------------------------------------
    # FIND TOP 200 BEST FOILS (BY MEAN L/D)
    # ------------------------------------------------------------

    print("\nComputing top 200 foils by average L/D...")

    df_valid = df.dropna(subset=["L_over_D"])

    if df_valid.empty:
        print("No valid aerodynamic results to rank.")
        return

    mean_ld = (
        df_valid
        .groupby("foil")["L_over_D"]
        .mean()
        .sort_values(ascending=False)
    )

    top_200 = mean_ld.head(200)

    top_200_df = top_200.reset_index()
    top_200_df.columns = ["foil", "mean_L_over_D"]


    # ------------------------------------------------------------
    # MERGE WITH LATENT PARAMETERS
    # ------------------------------------------------------------

    print("Loading latent parameters...")

    if not os.path.exists(LATENT_FILE):
        print("⚠️ Latent parameter file not found.")
        top_200_df.to_csv(TOP200_FILE, index=False)
        return

    latent_df = pd.read_csv(LATENT_FILE)

    # Rename filename column to foil
    latent_df = latent_df.rename(columns={"filename": "foil"})

    # Remove .png extension to match .dat names
    latent_df["foil"] = (
        latent_df["foil"]
        .astype(str)
        .str.replace(".png", "", regex=False)
    )

    top_200_df["foil"] = top_200_df["foil"].astype(str)

    # Merge
    top_200_with_params = pd.merge(
        top_200_df,
        latent_df,
        on="foil",
        how="left"
    )

    # Optional: warn if any params missing
    missing = top_200_with_params["foil"][
        top_200_with_params.isna().any(axis=1)
    ]

    if len(missing) > 0:
        print(f"⚠️ Warning: {len(missing)} foils missing latent params")

    # ------------------------------------------------------------
    # SAVE TOP 200
    # ------------------------------------------------------------

    print(f"Saving top 200 foils to '{TOP200_FILE}'...")
    top_200_with_params.to_csv(TOP200_FILE, index=False)

    print("\nTop 200 foils with latent parameters:")
    print(top_200_with_params.head())

    print("\nDone.")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    build_lookup_table()
