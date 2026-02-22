import torch
import torch.nn as nn
import numpy as np
import shap
import os
import pickle
import time
from tqdm import tqdm
import model_library as ml

# --- Configuration ---
MODELS_DIR = '../model/saved_models'
DATA_DIR = '../high_confidence_samples'
OUTPUT_DIR = 'deepSHAP_results'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32           # Hardware constraint (prevents OOM)
BACKGROUND_SIZE = 500    # background size 


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Running on device: {DEVICE}")
    print(f"PID: {os.getpid()}")

    if not os.path.exists(MODELS_DIR):
        print(f"Error: {MODELS_DIR} not found.")
        return

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]
    print(f"Found {len(model_files)} models to process.")

    for model_file in model_files:
        print(f"\n{'='*10} Processing {model_file} {'='*10}")
        
        # Load Model
        checkpoint_path = os.path.join(MODELS_DIR, model_file)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        
        target_cell_type = checkpoint['cell_type']
        gene_names = checkpoint['gene_names']
        num_genes = len(gene_names)
        
        # Locate Data
        safe_name = target_cell_type.replace(' ', '_').replace('/', '_')
        data_filename = f"{safe_name}_high_conf.npy"
        data_path = os.path.join(DATA_DIR, data_filename)

        if not os.path.exists(data_path):
            print(f"Skipping: Data file not found at {data_path}")
            continue
        
        # Load Data
        print(f"Loading data from {data_path}...")
        X_dense = np.load(data_path) 
        print(f"Data shape: {X_dense.shape}")

        # Initialize Model
        model = ml.BinaryClassifier(num_genes).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # --- DEEP EXPLATION SETUP ---

        # 1. Define Background
        # To avoid "artificial optimization," we use a large background (1000 samples).
        # We sample these from the input data itself.
        if X_dense.shape[0] > BACKGROUND_SIZE:
            bg_indices = np.random.choice(X_dense.shape[0], BACKGROUND_SIZE, replace=False)
            background_data = X_dense[bg_indices]
        else:
            print(f"Note: Dataset smaller than {BACKGROUND_SIZE}, using all samples as background.")
            background_data = X_dense
            
        background_tensor = torch.tensor(background_data, dtype=torch.float32).to(DEVICE)
        print(f"Background dataset size: {background_tensor.shape}")

        # 2. Initialize Explainer
        print("Initializing DeepExplainer...")
        explainer = shap.DeepExplainer(model, background_tensor)

        # 3. Compute in Batches
        print(f"Starting benchmark for {X_dense.shape[0]} cells...")
        start_time = time.time()
        
        shap_values_batches = []
        
        # Loop with progress bar
        num_samples = X_dense.shape[0]
        # tqdm wrapper for visibility
        for i in tqdm(range(0, num_samples, BATCH_SIZE), desc="Computing SHAP"):
            batch_data = X_dense[i : i + BATCH_SIZE]
            batch_tensor = torch.tensor(batch_data, dtype=torch.float32).to(DEVICE)
            
            # Compute SHAP
            # check_additivity=True ensures no numerical approximations are slipping by.
            batch_shap = explainer.shap_values(batch_tensor, check_additivity=True)
            
            if isinstance(batch_shap, list):
                batch_shap = batch_shap[0]
                
            shap_values_batches.append(batch_shap)

        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per cell: {total_time / num_samples:.4f} seconds")

        # Concatenate results
        shap_values = np.vstack(shap_values_batches)

        # 4. Save Results
        save_name = f"{safe_name}_deep_benchmark.pkl"
        save_full_path = os.path.join(OUTPUT_DIR, save_name)
        
        results = {
            'cell_type': target_cell_type,
            'gene_names': gene_names,
            'shap_values': shap_values, 
            'data_values': X_dense,
            'total_time_seconds': total_time,
            'background_size': BACKGROUND_SIZE
        }
        
        with open(save_full_path, 'wb') as f:
            pickle.dump(results, f)
            
        print(f"Saved results to {save_full_path}")

    print("\nBenchmark complete.")

if __name__ == "__main__":
    main()
