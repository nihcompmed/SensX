import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
IMG_NAMES = ['000276', '000375']
MODEL_NAMES = ['Smiling', 'Eyeglasses']

INPUT_DIR = 'perturbation_results'
OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_perturbation(img_name, model_name):
    """Plot perturbation analysis for one (image, model) case."""
    fname = os.path.join(INPUT_DIR, f'perturbation_{img_name}_{model_name}.npz')
    data = np.load(fname)
    
    k_values = data['k_values']
    top_k_qoi = data['top_k_qoi']  # (n_k, 1000)
    bot_k_qoi = data['bot_k_qoi']  # (n_k, 1000)
    baseline_qoi = data['baseline_qoi']
    
    # Compute statistics
    top_median = np.median(top_k_qoi, axis=1)
    top_q01 = np.percentile(top_k_qoi, 1, axis=1)
    top_q99 = np.percentile(top_k_qoi, 99, axis=1)
    
    bot_median = np.median(bot_k_qoi, axis=1)
    bot_q01 = np.percentile(bot_k_qoi, 1, axis=1)
    bot_q99 = np.percentile(bot_k_qoi, 99, axis=1)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Top-k (blue)
    ax.plot(k_values, top_median, '-', color='blue', linewidth=1.5, label='Top-$k$')
    ax.fill_between(k_values, top_q01, top_q99, color='blue', alpha=0.15)
    
    # Bot-k (red)
    ax.plot(k_values, bot_median, '-', color='red', linewidth=1.5, label='Bottom-$k$')
    ax.fill_between(k_values, bot_q01, bot_q99, color='red', alpha=0.15)
    
    # Baseline
    ax.axhline(baseline_qoi, color='black', linestyle='--', alpha=0.5, label='Baseline')
    
    #ax.set_xscale('log')
    ax.set_xlabel('$k$ (number of features perturbed)', fontsize=14)
    ax.set_ylabel('QOI (predicted probability)', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'perturbation_{img_name}_{model_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")

def main():
    for img_name in IMG_NAMES:
        for model_name in MODEL_NAMES:
            plot_perturbation(img_name, model_name)

if __name__ == '__main__':
    main()
