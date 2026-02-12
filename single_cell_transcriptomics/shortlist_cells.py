import torch
import torch.nn as nn
import anndata
import numpy as np
import scipy.sparse
import os
from tqdm import tqdm

import sys
sys.path.append('model')
import model as ml

# --- Configuration ---
H5AD_PATH = '4cb45d80-499a-48ae-a056-c71ac3552c94.h5ad'
MODELS_DIR = 'model/saved_models'
OUTPUT_DIR = 'high_confidence_samples'
CONFIDENCE_THRESHOLD = 0.99
SAMPLES_PER_TYPE = 1000
CELL_TYPE_COL = 'cell_type'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Running on device: {DEVICE}")

    # 1. Load Data
    print(f"Loading data from {H5AD_PATH}...")
    try:
        adata = anndata.read_h5ad(H5AD_PATH)
    except NameError:
        print("Error: 'anndata' missing. Run: pip install anndata")
        return

    # Ensure sparse format for efficiency
    if not scipy.sparse.issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)
    else:
        adata.X = adata.X.tocsr()

    # 2. Get list of trained models
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]
    if not model_files:
        print("No models found in saved_models/")
        return
    
    print(f"Found {len(model_files)} trained models.")

    # 3. Apply Gene Mask (Load from the first model)
    # Since the mask is global/constant for all models, we only need to load it once.
    first_checkpoint = torch.load(os.path.join(MODELS_DIR, model_files[0]), weights_only=False)
    gene_mask = first_checkpoint['gene_mask']
    
    print("Applying global gene mask...")
    adata = adata[:, gene_mask].copy()
    num_genes = adata.n_vars
    print(f"Data subsetted to {num_genes} non-constant genes.")

    # 4. Iterate over each model
    for model_file in model_files:
        print(f"\nProcessing {model_file}...")
        
        # Load Checkpoint
        checkpoint = torch.load(os.path.join(MODELS_DIR, model_file), weights_only=False)
        target_cell_type = checkpoint['cell_type']
        
        # Initialize Model
        model = ml.BinaryClassifier(num_genes).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Get indices of ALL cells belonging to this type (Ground Truth)
        # We only care about True Positives, so we start by selecting actual members.
        target_indices = np.where(adata.obs[CELL_TYPE_COL] == target_cell_type)[0]
        
        if len(target_indices) == 0:
            print(f"Warning: No cells of type '{target_cell_type}' found in adata. Skipping.")
            continue

        # Run Inference in Batches to find High Confidence Cells
        high_conf_indices = []
        batch_size = 1024
        
        with torch.no_grad():
            for i in range(0, len(target_indices), batch_size):
                batch_idx = target_indices[i : i + batch_size]
                
                # Fetch data (Sparse -> Dense)
                x_batch = adata.X[batch_idx].toarray().astype(np.float32)
                x_tensor = torch.tensor(x_batch).to(DEVICE)
                
                # Predict
                logits = model(x_tensor)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                
                # Filter: Confidence > 0.99
                # Since we pre-selected ground truth cells, these are all "Correct" predictions
                mask = probs > CONFIDENCE_THRESHOLD
                
                # Store the original indices of the passing cells
                high_conf_indices.extend(batch_idx[mask])

        num_found = len(high_conf_indices)
        print(f"  Found {num_found} cells with P > {CONFIDENCE_THRESHOLD}")

        if num_found == 0:
            print("  Skipping: No high-confidence cells found.")
            continue

        # Sample 1000 cells
        high_conf_indices = np.array(high_conf_indices)
        if num_found > SAMPLES_PER_TYPE:
            selected_indices = np.random.choice(high_conf_indices, SAMPLES_PER_TYPE, replace=False)
        else:
            print(f"  Note: Only {num_found} available (less than {SAMPLES_PER_TYPE}). Taking all.")
            selected_indices = high_conf_indices

        # Extract the Matrix (D genes)
        # Shape: (1000, D)
        final_matrix = adata.X[selected_indices].toarray().astype(np.float32)
        
        # Save
        # Sanitize filename
        safe_name = checkpoint['cell_type'].replace(' ', '_').replace('/', '_')
        save_path = os.path.join(OUTPUT_DIR, f"{safe_name}_high_conf.npy")
        np.save(save_path, final_matrix)
        
        print(f"  Saved matrix shape {final_matrix.shape} to {save_path}")

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()

