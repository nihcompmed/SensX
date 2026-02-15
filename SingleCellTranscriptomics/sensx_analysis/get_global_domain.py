import anndata
import numpy as np
import scipy.sparse
import torch
import os

# --- Configuration ---
H5AD_PATH = '../4cb45d80-499a-48ae-a056-c71ac3552c94.h5ad'
MODELS_DIR = '../model/saved_models'
OUTPUT_DIR = 'global_bounds' # Directory to save the npy files

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading data from {H5AD_PATH}...")
    try:
        adata = anndata.read_h5ad(H5AD_PATH)
    except NameError:
        print("Error: 'anndata' library not found.")
        return

    # 2. Retrieve Gene Mask from a Saved Model
    # We load the first available model to get the mask used during training
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError(f"No .pth model files found in {MODELS_DIR} to retrieve mask.")
    
    checkpoint_path = os.path.join(MODELS_DIR, model_files[0])
    print(f"Retrieving gene mask from: {checkpoint_path}")
    
    # Set weights_only=False because we are loading a dict with boolean arrays
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    gene_mask = checkpoint['gene_mask']
    
    # 3. Apply Mask to Data
    print(f"Total genes in h5ad: {adata.n_vars}")
    adata = adata[:, gene_mask].copy() # Subset to valid genes
    print(f"Genes after applying mask: {adata.n_vars}")
    
    # 4. Calculate Min and Max
    print("Calculating global min and max values (this may take a moment)...")
    
    if scipy.sparse.issparse(adata.X):
        # Convert to CSC format for faster column-wise operations
        print("Optimizing sparse matrix format (CSR -> CSC)...")
        X_csc = adata.X.tocsc()
        
        # Calculate min/max along axis 0 (columns/genes)
        # toarray() converts the result (1, n_genes) to dense, then flatten to 1D array
        global_lower = X_csc.min(axis=0).toarray().flatten()
        global_upper = X_csc.max(axis=0).toarray().flatten()
    else:
        # Dense matrix calculation
        global_lower = np.min(adata.X, axis=0)
        global_upper = np.max(adata.X, axis=0)

    # Ensure float32 for consistency with PyTorch
    global_lower = global_lower.astype(np.float32)
    global_upper = global_upper.astype(np.float32)

    # 5. Save to Disk
    lower_path = os.path.join(OUTPUT_DIR, 'global_lower.npy')
    upper_path = os.path.join(OUTPUT_DIR, 'global_upper.npy')
    
    np.save(lower_path, global_lower)
    np.save(upper_path, global_upper)
    
    print(f"\nSuccess!")
    print(f"Saved global_lower.npy ({global_lower.shape}) to {lower_path}")
    print(f"Saved global_upper.npy ({global_upper.shape}) to {upper_path}")
    
    # Validation check
    print(f"\nStats:")
    print(f"  Min value in lower bound: {global_lower.min()}")
    print(f"  Max value in upper bound: {global_upper.max()}")

if __name__ == "__main__":
    main()

