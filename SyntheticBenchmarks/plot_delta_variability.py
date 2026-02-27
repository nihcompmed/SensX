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


def load_deltas_stacked():
    """
    Load characteristic delta files and stack by (model, nw).
    Returns dict: {model: {nw: np.array of shape (n_runs, n_samples)}}
    """
    files = glob.glob(os.path.join(RESULTS_DIR, 'char_deltas_*.npy'))
    print(f"Found {len(files)} characteristic delta files.")

    raw = {m: {} for m in MODELS}

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
            deltas = np.load(f).flatten()
            if nw not in raw[model]:
                raw[model][nw] = {}
            raw[model][nw][run] = deltas
        except Exception as e:
            print(f"Failed to load {f}: {e}")

    stacked = {m: {} for m in MODELS}
    for model in MODELS:
        for nw in sorted(raw[model].keys()):
            runs = raw[model][nw]
            run_ids = sorted(runs.keys())
            arrays = [runs[r] for r in run_ids]
            lengths = set(len(a) for a in arrays)
            if len(lengths) > 1:
                print(f"WARNING: {model} nw={nw} has inconsistent sample counts: {lengths}")
                continue
            stacked[model][nw] = np.stack(arrays, axis=0)  # (n_runs, n_samples)
            print(f"  {model} nw={nw}: {stacked[model][nw].shape[0]} runs, {stacked[model][nw].shape[1]} samples")

    return stacked


def generate_delta_variability_plot(stacked):
    """
    One figure with four subplots (one per model).
    Each subplot: violin plot of per-sample std(delta) across runs, x-axis is n_w.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    FS_TITLE = 18
    FS_LABEL = 16
    FS_TICK = 13

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, model in enumerate(MODELS):
        ax = axes[i]
        model_data = stacked[model]
        if not model_data:
            ax.set_title(f"{MODEL_DISPLAY[model]} (no data)", fontsize=FS_TITLE)
            continue

        records = []
        for nw in sorted(model_data.keys()):
            arr = model_data[nw]  # (n_runs, n_samples)
            per_sample_std = np.std(arr, axis=0)  # (n_samples,)
            for s in per_sample_std:
                records.append({'nw': nw, 'Std': float(s)})

        plot_df = pd.DataFrame(records)
        nw_values = sorted(plot_df['nw'].unique())

        sns.violinplot(
            data=plot_df, x='nw', y='Std',
            order=nw_values, color='#1f77b4', alpha=0.7,
            inner='box', ax=ax, cut=0,
        )

        ax.set_title(f"{MODEL_DISPLAY[model]}", fontsize=FS_TITLE, fontweight='bold')
        ax.set_xlabel(r"$n_s = n_w$", fontsize=FS_LABEL)
        ax.set_ylabel(r"Std. dev. of $\delta$ across runs", fontsize=FS_LABEL)
        ax.tick_params(axis='both', which='major', labelsize=FS_TICK)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'delta_variability.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.savefig(fname.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")


if __name__ == "__main__":
    stacked = load_deltas_stacked()
    if all(len(v) == 0 for v in stacked.values()):
        print("No characteristic delta files found.")
    else:
        generate_delta_variability_plot(stacked)
