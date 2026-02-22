import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
RESULTS_DIR = 'results'
OUTPUT_DIR = 'figures'
MODELS = ['XOR', 'orange_skin', 'nonlinear_additive', 'switch']

MODEL_DISPLAY = {
    'XOR': 'XOR',
    'orange_skin': 'Orange Skin',
    'nonlinear_additive': 'Nonlinear Additive',
    'switch': 'Switch'
}


def load_characteristic_deltas():
    """
    Load characteristic delta files: char_deltas_{model}_nw{nw}_run{run}.npy
    Returns a DataFrame with columns: Model, nw, Run, Delta
    """
    files = glob.glob(os.path.join(RESULTS_DIR, 'char_deltas_*.npy'))
    print(f"Found {len(files)} characteristic delta files.")

    records = []
    for f in tqdm(files, desc="Loading deltas", unit="file"):
        base = os.path.basename(f)
        match = re.search(r'char_deltas_(.+)_nw(\d+)_run(\d+)\.npy', base)
        if not match:
            continue
        model = match.group(1)
        nw = int(match.group(2))
        run = int(match.group(3))
        if model not in MODELS:
            continue
        try:
            deltas = np.load(f)
            for d in deltas.flatten():
                records.append({
                    'Model': model,
                    'nw': nw,
                    'Run': run,
                    'Delta': float(d),
                })
        except Exception as e:
            print(f"Failed to load {f}: {e}")

    return pd.DataFrame(records)


def generate_characteristic_delta_plot(delta_df):
    """
    One figure with four subplots (one per model).
    Each subplot: violin plots of characteristic deltas, x-axis is n_w.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    FS_TITLE = 18
    FS_LABEL = 16
    FS_TICK = 13

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, model in enumerate(MODELS):
        ax = axes[i]
        model_data = delta_df[delta_df['Model'] == model]
        if model_data.empty:
            ax.set_title(f"{MODEL_DISPLAY[model]} (no data)", fontsize=FS_TITLE)
            continue

        nw_values = sorted(model_data['nw'].unique())

        sns.violinplot(
            data=model_data, x='nw', y='Delta',
            order=nw_values, color='#1f77b4', alpha=0.7,
            inner='box', ax=ax, cut=0,
        )

        ax.set_title(f"{MODEL_DISPLAY[model]}", fontsize=FS_TITLE, fontweight='bold')
        ax.set_xlabel(r"$n_w$", fontsize=FS_LABEL)
        ax.set_ylabel(r"Characteristic $\delta$", fontsize=FS_LABEL)
        ax.tick_params(axis='both', which='major', labelsize=FS_TICK)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'characteristic_deltas.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.savefig(fname.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")


if __name__ == "__main__":
    delta_df = load_characteristic_deltas()
    if delta_df.empty:
        print("No characteristic delta files found.")
    else:
        print(f"Loaded {len(delta_df)} delta values.")
        generate_characteristic_delta_plot(delta_df)
