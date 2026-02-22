import os
import glob
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
RESULTS_DIR = 'results'
MODELS = ['XOR', 'orange_skin', 'nonlinear_additive', 'switch']

GROUND_TRUTH_K = {
    'XOR': 2,
    'orange_skin': 4,
    'nonlinear_additive': 4,
    'switch': 5
}

MODEL_DISPLAY = {
    'XOR': 'XOR',
    'orange_skin': 'Orange Skin',
    'nonlinear_additive': 'Nonlinear Additive',
    'switch': 'Switch'
}

METHOD_DISPLAY = {
    'SensX': 'SensX',
    'SHAP': 'SHAP',
    'IG_zero': 'IG (zero)',
    'IG_mean': 'IG (mean)',
    'IG_random': 'IG (random)',
}

METHOD_COLORS = {
    'SensX': '#4477AA',
    'SHAP': '#EE6677',
    'IG_zero': '#228833',
    'IG_mean': '#CCBB44',
    'IG_random': '#AA3377',
}

ALL_METHODS = ['SensX', 'SHAP', 'IG_zero', 'IG_mean', 'IG_random']

OUTPUT_DIR = 'figures'


def parse_filename(filename):
    """Parse result filenames for SensX, SHAP, and IG."""
    base = os.path.basename(filename)

    # IG random: topk_acc_ig_{dataset}_ns{n_steps}_{baseline}_run{run}.npy
    match = re.search(r'topk_acc_ig_(.+)_ns(\d+)_(zero|mean|random)_run(\d+)\.npy', base)
    if match:
        method = f'IG_{match.group(3)}'
        return method, match.group(1), int(match.group(2)), int(match.group(4))

    # IG deterministic: topk_acc_ig_{dataset}_ns{n_steps}_{baseline}.npy
    match = re.search(r'topk_acc_ig_(.+)_ns(\d+)_(zero|mean)\.npy', base)
    if match:
        method = f'IG_{match.group(3)}'
        return method, match.group(1), int(match.group(2)), 0

    # SHAP: topk_acc_shap_{dataset}_ns{nsamples}_run{run}.npy
    match = re.search(r'topk_acc_shap_(.+)_ns(\d+)_run(\d+)\.npy', base)
    if match:
        return 'SHAP', match.group(1), int(match.group(2)), int(match.group(3))

    # SensX: topk_acc_{dataset}_nw{nw}_run{run}.npy
    match = re.search(r'topk_acc_(.+)_nw(\d+)_run(\d+)\.npy', base)
    if match:
        return 'SensX', match.group(1), int(match.group(2)), int(match.group(3))

    return None, None, None, None


def load_data():
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Directory '{RESULTS_DIR}' not found.")
        return {}

    files = glob.glob(os.path.join(RESULTS_DIR, '*.npy'))
    data_store = {m: {method: {} for method in ALL_METHODS} for m in MODELS}

    print(f"Found {len(files)} files. Loading...")

    for f in tqdm(files, desc="Scanning Files", unit="file"):
        method, model, param, run = parse_filename(f)
        if method is None or model not in MODELS:
            continue
        if method not in data_store[model]:
            continue
        if param not in data_store[model][method]:
            data_store[model][method][param] = {}
        try:
            data_store[model][method][param][run] = np.load(f)
        except Exception as e:
            print(f"Failed to load {f}: {e}")

    # Print summary
    for model in MODELS:
        print(f"\n{model}:")
        for method in ALL_METHODS:
            params = data_store[model][method]
            if params:
                for p, runs in sorted(params.items()):
                    print(f"  {method} param={p}: {len(runs)} runs")

    return data_store


def build_dataframe(data_store):
    """Build a single DataFrame with all methods, models, k values."""
    records = []

    for model in MODELS:
        max_k = GROUND_TRUTH_K[model]

        for method, params_dict in data_store[model].items():
            for param, runs_dict in params_dict.items():
                if not runs_dict:
                    continue
                run_ids = sorted(runs_dict.keys())
                stacked = np.array([runs_dict[r] for r in run_ids])
                actual_max_k = min(stacked.shape[2], max_k)

                for k_val in range(1, actual_max_k + 1):
                    # Mean accuracy across samples for each run
                    run_accuracies = np.mean(stacked[:, :, k_val - 1], axis=1) * 100
                    for rid, val in zip(run_ids, run_accuracies):
                        records.append({
                            'Model': model,
                            'Method': method,
                            'Hyperparameter': int(param),
                            'k': k_val,
                            'Run': rid,
                            'Accuracy (%)': val,
                        })

    return pd.DataFrame(records)


def find_best_hyperparams(df):
    """
    For each (Model, Method, k), find the hyperparameter with the best mean accuracy.
    Returns a DataFrame with columns: Model, Method, k, BestParam, MeanAcc, StdAcc
    """
    grouped = df.groupby(['Model', 'Method', 'k', 'Hyperparameter'])['Accuracy (%)'].agg(
        MeanAcc='mean', StdAcc='std'
    ).reset_index()

    # For each (Model, Method, k), pick the hyperparameter with highest mean
    idx = grouped.groupby(['Model', 'Method', 'k'])['MeanAcc'].idxmax()
    best = grouped.loc[idx].copy()
    best.rename(columns={'Hyperparameter': 'BestParam'}, inplace=True)

    # Fill NaN std (single-run cases) with 0
    best['StdAcc'] = best['StdAcc'].fillna(0)

    return best


def get_subplot_dims(n_plots):
    if n_plots == 1: return 1, 1
    if n_plots == 2: return 1, 2
    if n_plots <= 4: return 2, 2
    if n_plots <= 6: return 2, 3
    return math.ceil(n_plots / 3), 3


# =========================================================================
# BAR PLOTS
# =========================================================================

def generate_bar_plots(df, best_df):
    """
    Bar plot per model: x-axis is top-k, hue is method.
    Bars show best mean accuracy, error bars show std across runs.
    """
    print("\nGenerating bar plots...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    FS_LABEL = 20
    FS_TICK = 16
    FS_LEGEND = 14
    FS_TITLE = 22

    for model in MODELS:
        max_k = GROUND_TRUTH_K[model]
        model_best = best_df[best_df['Model'] == model].copy()
        if model_best.empty:
            continue

        fig, ax = plt.subplots(figsize=(max(8, max_k * 1.8), 6))

        # Prepare data for grouped bar plot
        k_values = list(range(1, max_k + 1))
        n_methods = len(ALL_METHODS)
        bar_width = 0.15
        x_positions = np.arange(len(k_values))

        for i, method in enumerate(ALL_METHODS):
            method_data = model_best[model_best['Method'] == method]
            if method_data.empty:
                continue

            means = []
            stds = []
            for k_val in k_values:
                row = method_data[method_data['k'] == k_val]
                if row.empty:
                    means.append(0)
                    stds.append(0)
                else:
                    means.append(row['MeanAcc'].values[0])
                    stds.append(row['StdAcc'].values[0])

            offset = (i - n_methods / 2 + 0.5) * bar_width
            bars = ax.bar(
                x_positions + offset, means, bar_width,
                yerr=stds, capsize=3,
                label=METHOD_DISPLAY[method],
                color=METHOD_COLORS[method],
                edgecolor='black', linewidth=0.5,
                error_kw={'linewidth': 1.2},
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'Top-{k}' for k in k_values], fontsize=FS_TICK)
        ax.set_ylabel("Accuracy (%)", fontsize=FS_LABEL)
        ax.set_ylim(0, 105)
        ax.tick_params(axis='y', labelsize=FS_TICK)
        ax.legend(fontsize=FS_LEGEND, frameon=True, loc='best')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        ax.set_title(f"{MODEL_DISPLAY[model]}", fontsize=FS_TITLE, fontweight='bold')

        plt.tight_layout()
        fname = os.path.join(OUTPUT_DIR, f'barplot_{model}.png')
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.savefig(fname.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  Saved {fname}")


# =========================================================================
# CONVERGENCE PLOTS
# =========================================================================

def generate_sensx_shap_convergence(df, best_df):
    """
    Per model: one figure with K subplots.
    Each subplot shows SensX and SHAP accuracy vs hyperparameter with star at best.
    """
    print("\nGenerating SensX + SHAP convergence figures...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    FS_TITLE = 18
    FS_LABEL = 16
    FS_TICK = 13
    FS_LEGEND = 12
    STAR_SIZE = 300

    for model in MODELS:
        max_k = GROUND_TRUTH_K[model]
        model_df = df[df['Model'] == model]
        model_best = best_df[best_df['Model'] == model]

        nrows, ncols = get_subplot_dims(max_k)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        if max_k == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for k_idx, k_val in enumerate(range(1, max_k + 1)):
            ax = axes[k_idx]
            subset_k = model_df[model_df['k'] == k_val]

            for method in ['SensX', 'SHAP']:
                method_data = subset_k[subset_k['Method'] == method]
                if method_data.empty:
                    continue

                sns.lineplot(
                    data=method_data, x='Hyperparameter', y='Accuracy (%)',
                    label=METHOD_DISPLAY[method], color=METHOD_COLORS[method],
                    markers=True, markersize=8, dashes=False,
                    err_style='band', errorbar='sd', ax=ax,
                )

                # Star at best hyperparameter
                best_row = model_best[
                    (model_best['Method'] == method) & (model_best['k'] == k_val)
                ]
                if not best_row.empty:
                    bx = best_row['BestParam'].values[0]
                    by = best_row['MeanAcc'].values[0]
                    ax.scatter(bx, by, marker='*', s=STAR_SIZE,
                               color=METHOD_COLORS[method], edgecolors='black',
                               linewidths=0.8, zorder=10)

            ax.set_xscale('log')
            ax.set_title(f"Top-{k_val}", fontsize=FS_TITLE, fontweight='bold')
            ax.set_xlabel("Hyperparameter", fontsize=FS_LABEL)
            ax.set_ylabel("Accuracy (%)", fontsize=FS_LABEL)
            ax.tick_params(axis='both', which='major', labelsize=FS_TICK)
            ax.set_ylim(-5, 105)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(fontsize=FS_LEGEND, frameon=True, loc='best')

        for i in range(max_k, len(axes)):
            axes[i].axis('off')

        fig.suptitle(f"{MODEL_DISPLAY[model]} — SensX & SHAP Convergence",
                     fontsize=FS_TITLE + 2, fontweight='bold', y=1.02)
        plt.tight_layout()
        fname = os.path.join(OUTPUT_DIR, f'convergence_sensx_shap_{model}.png')
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.savefig(fname.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  Saved {fname}")


def generate_ig_convergence(df, best_df):
    """
    Per model: one figure with K subplots.
    Each subplot shows IG_zero, IG_mean, IG_random accuracy vs n_steps with star at best.
    """
    print("\nGenerating IG convergence figures...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    FS_TITLE = 18
    FS_LABEL = 16
    FS_TICK = 13
    FS_LEGEND = 12
    STAR_SIZE = 300

    ig_methods = ['IG_zero', 'IG_mean', 'IG_random']

    for model in MODELS:
        max_k = GROUND_TRUTH_K[model]
        model_df = df[df['Model'] == model]
        model_best = best_df[best_df['Model'] == model]

        nrows, ncols = get_subplot_dims(max_k)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        if max_k == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for k_idx, k_val in enumerate(range(1, max_k + 1)):
            ax = axes[k_idx]
            subset_k = model_df[model_df['k'] == k_val]

            for method in ig_methods:
                method_data = subset_k[subset_k['Method'] == method]
                if method_data.empty:
                    continue

                sns.lineplot(
                    data=method_data, x='Hyperparameter', y='Accuracy (%)',
                    label=METHOD_DISPLAY[method], color=METHOD_COLORS[method],
                    markers=True, markersize=8, dashes=False,
                    err_style='band', errorbar='sd', ax=ax,
                )

                # Star at best hyperparameter
                best_row = model_best[
                    (model_best['Method'] == method) & (model_best['k'] == k_val)
                ]
                if not best_row.empty:
                    bx = best_row['BestParam'].values[0]
                    by = best_row['MeanAcc'].values[0]
                    ax.scatter(bx, by, marker='*', s=STAR_SIZE,
                               color=METHOD_COLORS[method], edgecolors='black',
                               linewidths=0.8, zorder=10)

            ax.set_title(f"Top-{k_val}", fontsize=FS_TITLE, fontweight='bold')
            ax.set_xlabel("n_steps", fontsize=FS_LABEL)
            ax.set_ylabel("Accuracy (%)", fontsize=FS_LABEL)
            ax.tick_params(axis='both', which='major', labelsize=FS_TICK)
            ax.set_ylim(-5, 105)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(fontsize=FS_LEGEND, frameon=True, loc='best')

        for i in range(max_k, len(axes)):
            axes[i].axis('off')

        fig.suptitle(f"{MODEL_DISPLAY[model]} — IG Convergence",
                     fontsize=FS_TITLE + 2, fontweight='bold', y=1.02)
        plt.tight_layout()
        fname = os.path.join(OUTPUT_DIR, f'convergence_ig_{model}.png')
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.savefig(fname.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  Saved {fname}")


# =========================================================================
# SUMMARY TABLE
# =========================================================================

def generate_summary_table(best_df):
    """Print a summary table: best accuracy per method at each k."""
    print("\n--- Summary: Best accuracy per method at each k ---")
    print(f"{'Model':<25} {'k':<5} {'Method':<15} {'Best Acc (%)':<15} {'± Std':<10} {'Hyperparam':<12}")
    print("-" * 82)

    for model in MODELS:
        max_k = GROUND_TRUTH_K[model]
        model_best = best_df[best_df['Model'] == model]

        for k_val in range(1, max_k + 1):
            k_best = model_best[model_best['k'] == k_val]

            for method in ALL_METHODS:
                row = k_best[k_best['Method'] == method]
                if row.empty:
                    continue
                acc = row['MeanAcc'].values[0]
                std = row['StdAcc'].values[0]
                param = row['BestParam'].values[0]
                print(f"{model:<25} {k_val:<5} {METHOD_DISPLAY[method]:<15} {acc:<15.1f} {std:<10.1f} {param:<12}")

            print()
        print("-" * 82)


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_data()
    df = build_dataframe(data)

    if df.empty:
        print("No data found.")
    else:
        best_df = find_best_hyperparams(df)

        # Bar plots (switch = main text, others = supplemental)
        generate_bar_plots(df, best_df)

        # Convergence plots (all supplemental)
        generate_sensx_shap_convergence(df, best_df)
        generate_ig_convergence(df, best_df)

        # Summary table
        generate_summary_table(best_df)

        print(f"\nAll figures saved to '{OUTPUT_DIR}/'")
