"""
tools/build_lookup_table.py

PURPOSE:
  Build a lookup table of ALL 1647 training foils evaluated through the
  actual decoder + NeuralFoil pipeline at target operating conditions.

  This replaces the random seed search in nom_driver.py with a smart
  baseline: start optimization from the best-performing training foil.

ACTION ITEMS:
  - "Make lookup table of all foils and different variations of reynolds
     number and angle of attack, find best airfoil, use that as baseline"
  - "Use lookup table earlier in nom_driver.py instead of seeds"
  - "Alpha 6 is too harsh" → use alpha=4 or 5
  - "5e5 also for reynolds number" → use Re=5e5

HOW TO RUN:
  python tools/build_lookup_table.py

OUTPUTS:
  outputs/lookup_table_alpha4_Re5e5.csv       - all 1647 foils ranked
  outputs/top_100_foils_alpha4_Re5e5.csv      - top 100 only
  outputs/best_baseline_foil_alpha4_Re5e5.json - best single foil for nom_driver

WHAT IT DOES:
  1. Load all 1647 latent parameter vectors from training CSV
  2. For each foil, evaluate through: latent → decoder → coords → NeuralFoil
  3. Record CL, CD, L/D at target conditions (alpha=4, Re=5e5)
  4. Sort by L/D (or CD/CL) to find best baseline
  5. Save lookup table for future experiments

WHY THIS IS BETTER THAN RANDOM SEED SEARCH:
  - Guaranteed to find actual best training foil
  - Tests decoder's ability to reproduce training data
  - Reusable for different alpha/Re experiments
  - Much smarter starting point for NOM refinement
"""

from __future__ import annotations
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from pipeline.talarai_pipeline import TalarAIPipeline
except ModuleNotFoundError:
    from talarai_pipeline import TalarAIPipeline


# ============================================================================
# CONFIGURATION
# ============================================================================

# Input files
LATENT_CSV = PROJECT_ROOT / "data" / "airfoil_latent_params_6.csv"

# Output files
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Target operating conditions (from prof feedback)
ALPHA = 1.0   # Prof: "6 is too harsh" → use 4 or 5
RE = 440000      # Prof: "5e5 also for reynolds number"

# Output filenames include conditions for clarity
LOOKUP_FILE = OUTPUT_DIR / f"lookup_table_alpha{ALPHA:.0f}_Re{RE:.0e}.csv"
TOP100_FILE = OUTPUT_DIR / f"top_100_foils_alpha{ALPHA:.0f}_Re{RE:.0e}.csv"
BEST_FILE   = OUTPUT_DIR / f"best_baseline_foil_alpha{ALPHA:.0f}_Re{RE:.0e}.json"


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def build_lookup_table():
    """
    Evaluate all training foils through decoder + NeuralFoil pipeline.
    """
    
    print("=" * 70)
    print("LOOKUP TABLE BUILDER")
    print("=" * 70)
    print(f"Target conditions: alpha={ALPHA}°, Re={RE:.0e}")
    print(f"Input CSV: {LATENT_CSV}")
    print(f"Output: {LOOKUP_FILE}")
    print("=" * 70)
    print()
    
    # ------------------------------------------------------------------------
    # Step 1: Load latent parameters
    # ------------------------------------------------------------------------
    
    if not LATENT_CSV.exists():
        raise FileNotFoundError(f"Latent CSV not found: {LATENT_CSV}")
    
    df = pd.read_csv(LATENT_CSV)
    
    # Extract latent columns (p1..p6)
    latent_cols = [c for c in df.columns if c.lower().startswith('p')]
    if len(latent_cols) != 6:
        raise ValueError(f"Expected 6 latent columns, found {len(latent_cols)}")
    
    all_latents = df[latent_cols].values.astype(np.float32)
    filenames = df['filename'].tolist() if 'filename' in df.columns else [f"foil_{i}" for i in range(len(df))]
    
    n_foils = len(all_latents)
    print(f"Loaded {n_foils} foils from {LATENT_CSV.name}")
    print()
    
    # ------------------------------------------------------------------------
    # Step 2: Initialize pipeline (decoder + NeuralFoil)
    # ------------------------------------------------------------------------
    
    print("Initializing TalarAIPipeline (loading decoder model)...")
    try:
        pipeline = TalarAIPipeline()
        print(f"✓ Pipeline ready (decoder: {pipeline.decoder_path.name})")
    except Exception as e:
        print(f"✗ Failed to load pipeline: {e}")
        raise
    print()
    
    # ------------------------------------------------------------------------
    # Step 3: Evaluate each foil
    # ------------------------------------------------------------------------
    
    print(f"Evaluating {n_foils} foils at alpha={ALPHA}°, Re={RE:.0e}...")
    print("(This may take 5-10 minutes depending on your hardware)")
    print()
    
    results = []
    failed = []
    
    for i, (latent, filename) in enumerate(zip(all_latents, filenames)):
        
        # Progress indicator every 100 foils
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1:4d}/{n_foils}] Evaluating {filename}...")
        
        try:
            # Evaluate through full pipeline: latent → decoder → coords → NeuralFoil
            aero = pipeline.eval_latent_with_neuralfoil(
                latent_vec=latent,
                alpha=ALPHA,
                Re=RE,
                model_size="xlarge",
                debug=False,
            )
            
            CL = float(aero.get("CL", np.nan))
            CD = float(aero.get("CD", np.nan))
            
            # Compute performance metrics
            if np.isfinite(CL) and np.isfinite(CD) and CL > 0 and CD > 0:
                L_over_D = CL / CD
                CD_over_CL = CD / CL
            else:
                L_over_D = np.nan
                CD_over_CL = np.nan
            
            results.append({
                'filename': filename,
                'CL': CL,
                'CD': CD,
                'L_over_D': L_over_D,
                'CD_over_CL': CD_over_CL,
                'p1': float(latent[0]),
                'p2': float(latent[1]),
                'p3': float(latent[2]),
                'p4': float(latent[3]),
                'p5': float(latent[4]),
                'p6': float(latent[5]),
            })
            
        except Exception as e:
            failed.append({'filename': filename, 'error': str(e)})
            results.append({
                'filename': filename,
                'CL': np.nan,
                'CD': np.nan,
                'L_over_D': np.nan,
                'CD_over_CL': np.nan,
                'p1': float(latent[0]),
                'p2': float(latent[1]),
                'p3': float(latent[2]),
                'p4': float(latent[3]),
                'p5': float(latent[4]),
                'p6': float(latent[5]),
            })
    
    print()
    print(f"✓ Evaluation complete!")
    print(f"  Successful: {n_foils - len(failed)}/{n_foils}")
    print(f"  Failed:     {len(failed)}/{n_foils}")
    print()
    
    # ------------------------------------------------------------------------
    # Step 4: Save full lookup table
    # ------------------------------------------------------------------------
    
    df_lookup = pd.DataFrame(results)
    
    # Sort by L/D (best first) - NaN will be at the end
    df_lookup = df_lookup.sort_values('L_over_D', ascending=False, na_position='last')
    
    print(f"Saving full lookup table to: {LOOKUP_FILE}")
    df_lookup.to_csv(LOOKUP_FILE, index=False)
    print(f"✓ Saved {len(df_lookup)} rows")
    print()
    
    # ------------------------------------------------------------------------
    # Step 5: Save top 100 foils
    # ------------------------------------------------------------------------
    
    df_valid = df_lookup.dropna(subset=['L_over_D'])
    
    if df_valid.empty:
        print("⚠️  WARNING: No valid foils found (all returned NaN)")
        return
    
    df_top100 = df_valid.head(100)
    
    print(f"Saving top 100 foils to: {TOP100_FILE}")
    df_top100.to_csv(TOP100_FILE, index=False)
    print(f"✓ Saved top 100 foils")
    print()
    
    # ------------------------------------------------------------------------
    # Step 6: Save best single foil (for nom_driver.py)
    # ------------------------------------------------------------------------
    
    best_row = df_valid.iloc[0]
    
    best_foil = {
        'filename': best_row['filename'],
        'alpha': ALPHA,
        'Re': RE,
        'CL': float(best_row['CL']),
        'CD': float(best_row['CD']),
        'L_over_D': float(best_row['L_over_D']),
        'CD_over_CL': float(best_row['CD_over_CL']),
        'latent': [
            float(best_row['p1']),
            float(best_row['p2']),
            float(best_row['p3']),
            float(best_row['p4']),
            float(best_row['p5']),
            float(best_row['p6']),
        ],
    }
    
    print(f"Saving best baseline foil to: {BEST_FILE}")
    with open(BEST_FILE, 'w') as f:
        json.dump(best_foil, f, indent=2)
    print(f"✓ Saved best foil")
    print()
    
    # ------------------------------------------------------------------------
    # Step 7: Print summary
    # ------------------------------------------------------------------------
    
    print("=" * 70)
    print("LOOKUP TABLE COMPLETE")
    print("=" * 70)
    print()
    print("BEST BASELINE FOIL:")
    print(f"  Filename:  {best_foil['filename']}")
    print(f"  CL:        {best_foil['CL']:.4f}")
    print(f"  CD:        {best_foil['CD']:.6f}")
    print(f"  L/D:       {best_foil['L_over_D']:.2f}")
    print(f"  CD/CL:     {best_foil['CD_over_CL']:.6f}")
    print()
    print("TOP 10 FOILS:")
    print(df_top100[['filename', 'CL', 'CD', 'L_over_D']].head(10).to_string(index=False))
    print()
    print("=" * 70)
    print()
    print("NEXT STEPS:")
    print("  1. Check the top 10 foils above - do they look reasonable?")
    print("  2. Update nom_driver.py to use best_baseline_foil_*.json")
    print("  3. Run nom_driver.py - it will start from this proven baseline")
    print()
    print("=" * 70)


if __name__ == "__main__":
    build_lookup_table()