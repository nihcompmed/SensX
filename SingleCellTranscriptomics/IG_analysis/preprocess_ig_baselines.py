"""
Preprocessing script for IG baselines.
Run once before ig_singlecell.py.

Loads the h5ad file, reconstructs the training split (random_state=42),
and saves:
  - training_mean.npy: mean vector across all training cells
  - random_baselines/random_baseline_run{1..100}.npy: 1000 random training
    samples per run (matching the number of high-confidence cells per cell type)

Usage:
    python3 preprocess_ig_baselines.py
"""

import numpy as np
import anndata
import scipy.sparse
from sklearn.model_selection import train_test_split
import os

# --- Configuration (must match train_models.py) ---
H5AD_PATH = '../4cb45d80-499a-48ae-a056-c71ac3552c94.h5ad'
MIN_CELLS_PER_TYPE = 10000
CELL_TYPE_COL = 'cell_type'
N_RANDOM_RUNS = 100
SAMPLES_PER_RUN = 1000  # matches high-confidence cell count per type
OUTPUT_DIR = 'ig_baselines'

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'random_baselines'), exist_ok=True)

    # --- Load and filter (same as train_models.py) ---
    print(f"Loading {H5AD_PATH}...")
    adata = anndata.read_h5ad(H5AD_PATH)

    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.tocsr()
        mean_sq = np.array(adata.X.power(2).mean(axis=0)).flatten()
        mean = np.array(adata.X.mean(axis=0)).flatten()
        var = mean_sq - mean**2
    else:
        var = np.var(adata.X, axis=0)

    non_constant_mask = (var > 1e-6)
    adata = adata[:, non_constant_mask].copy()
    print(f"After gene filter: {adata.shape}")

    counts = adata.obs[CELL_TYPE_COL].value_counts()
    valid_types = counts[counts >= MIN_CELLS_PER_TYPE].index.tolist()
    adata = adata[adata.obs[CELL_TYPE_COL].isin(valid_types)].copy()
    print(f"After cell type filter: {adata.shape}")

    # --- Reconstruct training split (random_state=42) ---
    all_indices = np.arange(adata.n_obs)
    all_labels = adata.obs[CELL_TYPE_COL].values

    train_idx, _, _, _ = train_test_split(
        all_indices, all_labels,
        test_size=0.3,
        stratify=all_labels,
        random_state=42
    )

    print(f"Training indices: {len(train_idx)}")

    # --- Extract training data ---
    X_sparse = adata.X
    if scipy.sparse.issparse(X_sparse):
        X_train = X_sparse[train_idx].toarray().astype(np.float32)
    else:
        X_train = np.array(X_sparse[train_idx], dtype=np.float32)

    print(f"Training data shape: {X_train.shape}")

    # --- Save mean ---
    mean_path = os.path.join(OUTPUT_DIR, 'training_mean.npy')
    np.save(mean_path, X_train.mean(axis=0))
    print(f"Saved training mean to {mean_path}")

    # --- Save random subsets ---
    n_train = len(X_train)
    for run_id in range(1, N_RANDOM_RUNS + 1):
        rng = np.random.default_rng(seed=run_id)
        idxs = rng.choice(n_train, size=SAMPLES_PER_RUN, replace=True)
        subset = X_train[idxs]
        save_path = os.path.join(OUTPUT_DIR, 'random_baselines', f'random_baseline_run{run_id}.npy')
        np.save(save_path, subset)

    print(f"Saved {N_RANDOM_RUNS} random baseline subsets ({SAMPLES_PER_RUN} samples each)")

    del adata, X_train
    print("Done.")


if __name__ == "__main__":
    main()
