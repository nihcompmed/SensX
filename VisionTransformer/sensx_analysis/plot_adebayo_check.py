import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# --- Configuration ---
IMG_NAME = '000276'
MODEL_NAME = 'Eyeglasses'
N_W = 20
RANK_THRESHOLD = 1000

SENSITIVITY_DIR = 'sensitivity'
ADEBAYO_DIR = 'adebayo_analysis/sensitivity'
OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LEVELS = {
    'base': {
        'dir': SENSITIVITY_DIR,
        'pattern': f'sensx_{IMG_NAME}_{MODEL_NAME}_nw{N_W}_batch*.npy',
        'label': 'Trained Model'
    },
    'level_1': {
        'dir': ADEBAYO_DIR,
        'pattern': f'sensx_{IMG_NAME}_{MODEL_NAME}_level_1_block11_nw{N_W}_batch*.npy',
        #'label': 'Level 1 (Block 11)'
        'label': 'Block 11 randomized'
    },
    'level_2': {
        'dir': ADEBAYO_DIR,
        'pattern': f'sensx_{IMG_NAME}_{MODEL_NAME}_level_2_blocks8to11_nw{N_W}_batch*.npy',
        #'label': 'Level 2 (Blocks 8–11)'
        'label': 'Blocks 8–11 randomized'
    }
}

def load_and_average(directory, pattern):
    """Load all batch files matching pattern and average."""
    batch_files = sorted(glob.glob(os.path.join(directory, pattern)))
    if len(batch_files) == 0:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    print(f"  Found {len(batch_files)} batch files")
    
    arrays = [np.load(f) for f in batch_files]
    stacked = np.stack(arrays, axis=0)
    averaged = np.mean(stacked, axis=0)
    return averaged.squeeze().flatten()

def compute_ranks(values):
    """Rank descending: rank 1 = largest value."""
    return np.argsort(np.argsort(-values)) + 1

def plot_rank_scatter(base_ranks, other_ranks, other_label, threshold, save_path):
    """Plot rank scatter for features in top-threshold of either map."""
    mask = (base_ranks <= threshold) | (other_ranks <= threshold)
    r_base = base_ranks[mask]
    r_other = other_ranks[mask]
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    ax.scatter(r_base, r_other, alpha=0.5, s=10, color='tab:red', edgecolors='none')
    
    max_r = max(np.max(r_base), np.max(r_other))
    ax.plot([1, max_r], [1, max_r], ls='--', color='black', lw=1.5, label='Identity')
    
    ax.set_xlabel(f'SensX ranks (Trained Model)', fontsize=20)
    ax.set_ylabel(f'SensX ranks ({other_label})', fontsize=20)
    ax.tick_params(axis='both', labelsize=16)
    ax.tick_params(axis='x', rotation=15)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {save_path}")

def main():
    # Load all maps
    print("Loading base model sensitivity...")
    base_flat = load_and_average(LEVELS['base']['dir'], LEVELS['base']['pattern'])
    base_ranks = compute_ranks(base_flat)
    
    for level_key in ['level_1', 'level_2']:
        level = LEVELS[level_key]
        print(f"\nLoading {level['label']} sensitivity...")
        level_flat = load_and_average(level['dir'], level['pattern'])
        level_ranks = compute_ranks(level_flat)
        
        fname = f'adebayo_rank_scatter_{IMG_NAME}_{MODEL_NAME}_{level_key}.png'
        plot_rank_scatter(base_ranks, level_ranks, level['label'], 
                         RANK_THRESHOLD, os.path.join(OUTPUT_DIR, fname))

if __name__ == '__main__':
    main()
