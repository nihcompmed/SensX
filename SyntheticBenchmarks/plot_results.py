import os
import glob
import re
import math
import numpy as np
import matplotlib.pyplot as plt
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

def parse_filename(filename):
    base = os.path.basename(filename)
    if 'shap' in base:
        method = 'SHAP'
        match = re.search(r'topk_acc_shap_(.+)_ns(\d+)_run(\d+)\.npy', base)
        if match:
            return method, match.group(1), int(match.group(2)), int(match.group(3))
    else:
        method = 'SensX'
        match = re.search(r'topk_acc_(.+)_nw(\d+)_run(\d+)\.npy', base)
        if match:
            return method, match.group(1), int(match.group(2)), int(match.group(3))
    return None, None, None, None

def load_data():
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Directory '{RESULTS_DIR}' not found.")
        return {}

    files = glob.glob(os.path.join(RESULTS_DIR, '*.npy'))
    data_store = {m: {'SensX': {}, 'SHAP': {}} for m in MODELS}
    
    print(f"Found {len(files)} files. Loading...")
    
    for f in tqdm(files, desc="Scanning Files", unit="file"):
        method, model, param, run = parse_filename(f)
        if method is None or model not in MODELS:
            continue
        if param not in data_store[model][method]:
            data_store[model][method][param] = {}
        try:
            data_store[model][method][param][run] = np.load(f)
        except Exception as e:
            print(f"Failed to load {f}: {e}")
            
    return data_store

def get_subplot_dims(n_plots):
    if n_plots == 1: return 1, 1
    if n_plots == 2: return 1, 2
    if n_plots <= 4: return 2, 2
    if n_plots <= 6: return 2, 3
    return math.ceil(n_plots / 3), 3

def generate_per_model_plots(data_store):
    print("\nGenerating per-model accuracy plots...")
    
    for model in tqdm(MODELS, desc="Processing Models"):
        max_k = GROUND_TRUTH_K[model]
        acc_records = []
        
        # Extract data for all K up to max_k
        for method in ['SensX', 'SHAP']:
            params = sorted(data_store[model][method].keys())
            
            for param in params:
                runs_dict = data_store[model][method][param]
                if not runs_dict: continue
                
                run_ids = sorted(runs_dict.keys())
                stacked = np.array([runs_dict[r] for r in run_ids]) 
                
                if stacked.shape[2] < max_k:
                    actual_max_k = stacked.shape[2]
                else:
                    actual_max_k = max_k

                for k_val in range(1, actual_max_k + 1):
                    k_idx = k_val - 1
                    data_k = stacked[:, :, k_idx]
                    
                    # Metric: Accuracy (mean over samples) -> One value per run
                    run_accuracies = np.mean(data_k, axis=1) * 100
                    for val in run_accuracies:
                        acc_records.append({
                            'Method': method,
                            'Hyperparameter': param,
                            'k': k_val,
                            'Accuracy (%)': val
                        })

        df_acc = pd.DataFrame(acc_records)
        
        if df_acc.empty:
            continue

        df_acc['Hyperparameter'] = pd.to_numeric(df_acc['Hyperparameter'])

        # --- PLOT: ACCURACY ---
        nrows, ncols = get_subplot_dims(max_k)
        fig_acc, axes_acc = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
        if max_k > 1: axes_acc = axes_acc.flatten()
        else: axes_acc = [axes_acc]
        
        for k_val in range(1, max_k + 1):
            ax = axes_acc[k_val-1]
            subset = df_acc[df_acc['k'] == k_val]
            if not subset.empty:
                sns.lineplot(data=subset, x='Hyperparameter', y='Accuracy (%)', hue='Method',
                             style='Method', markers=True, dashes=False, err_style='band', errorbar='sd',
                             ax=ax, palette={'SensX': 'blue', 'SHAP': 'orange'})
            ax.set_title(f"Top-{k_val} Accuracy")
            ax.set_xlabel("Compute Budget (Coalitions / Walks)")
            ax.set_ylabel("Accuracy (%)")
            
            # --- FIXED Y-AXIS ---
            ax.set_ylim(-5, 105) # Fixed range from -5 to 105 for clear visibility
            ax.grid(True, linestyle='--', alpha=0.5)

        for i in range(max_k, len(axes_acc)): axes_acc[i].axis('off')
            
        plt.suptitle(f"{model}: Accuracy Convergence vs Compute Budget", fontsize=16)
        plt.tight_layout()
        plt.savefig(f'convergence_accuracy_{model}.png', dpi=300)
        plt.close()

if __name__ == "__main__":
    data = load_data()
    generate_per_model_plots(data)
