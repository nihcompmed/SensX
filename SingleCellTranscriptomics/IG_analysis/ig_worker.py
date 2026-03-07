import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import time
from tqdm import tqdm
import re
import sys
from captum.attr import IntegratedGradients

sys.path.append('../model')
import model_library as ml

# ---------------------------------------------------------
# Command Line Arguments
# Expected: python3 ig_singlecell.py <baseline_type> [run_number]
#
# baseline_type:
#   - zero:             zero vector (deterministic)
#   - mean_train:       mean of training data (deterministic)
#   - mean_shortlist:   mean of shortlisted high-confidence cells (deterministic)
#   - random_train:     random samples from training data (sampling-based)
#   - random_shortlist: random samples from shortlisted cells (sampling-based)
#
# For random_* baselines, provide run_number as 2nd argument.
# ---------------------------------------------------------
baseline_type = sys.argv[1]
run_number = int(sys.argv[2]) if len(sys.argv) > 2 else 0

# --- Configuration ---
MODELS_DIR = '../model/saved_models'
DATA_DIR = '../high_confidence_samples'
BASELINES_DIR = 'ig_baselines'
OUTPUT_DIR = f'IG_results_{baseline_type}'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
N_STEPS = 200


def sanitize_filename(name):
    return re.sub(r'[^\w\-_\. ]', '_', name).replace(' ', '_')


def model_forward(model):
    """Wrap model to return sigmoid probability (QOI)."""
    def forward_fn(x):
        logits = model(x).squeeze(-1)
        return torch.sigmoid(logits)
    return forward_fn


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Running on device: {DEVICE}")
    print(f"Baseline type: {baseline_type}")
    if baseline_type.startswith("random"):
        print(f"Run number: {run_number}")
    print(f"n_steps: {N_STEPS}")

    if not os.path.exists(MODELS_DIR):
        print(f"Error: {MODELS_DIR} not found.")
        return

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]
    print(f"Found {len(model_files)} models to process.")

    # --- Load precomputed baselines if needed ---
    X_train_mean = None
    X_train_random = None

    if baseline_type == "mean_train":
        mean_path = os.path.join(BASELINES_DIR, 'training_mean.npy')
        X_train_mean = np.load(mean_path)
        print(f"Loaded training mean from {mean_path}")

    elif baseline_type == "random_train":
        random_path = os.path.join(BASELINES_DIR, 'random_baselines', f'random_baseline_run{run_number}.npy')
        X_train_random = np.load(random_path)
        print(f"Loaded random baseline from {random_path}: {X_train_random.shape}")

    for model_file in model_files:
        print(f"\n{'='*10} Processing {model_file} {'='*10}")

        # Load Model
        checkpoint_path = os.path.join(MODELS_DIR, model_file)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

        target_cell_type = checkpoint['cell_type']
        gene_names = checkpoint['gene_names']
        num_genes = len(gene_names)

        # Locate Data
        safe_name = sanitize_filename(target_cell_type).replace(' ', '_').replace('/', '_')
        data_filename = f"{safe_name}_high_conf.npy"
        data_path = os.path.join(DATA_DIR, data_filename)

        # Output filename
        if baseline_type.startswith("random"):
            save_name = f"{safe_name}_ig_{baseline_type}_run{run_number}.pkl"
        else:
            save_name = f"{safe_name}_ig_{baseline_type}.pkl"
        save_full_path = os.path.join(OUTPUT_DIR, save_name)

        if os.path.isfile(save_full_path):
            print(f"Skipping: {save_full_path} already exists.")
            continue

        if not os.path.exists(data_path):
            print(f"Skipping: Data file not found at {data_path}")
            continue

        # Load high-confidence cell data
        print(f"Loading data from {data_path}...")
        X_dense = np.load(data_path)
        num_samples = X_dense.shape[0]
        print(f"Data shape: {X_dense.shape}")

        # Initialize Model
        model = ml.BinaryClassifier(num_genes).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # --- Construct Baseline ---
        # "shortlist" variants use high-confidence cells (same source as DeepSHAP background).
        # "train" variants use reconstructed training data (what the model saw during training).

        if baseline_type == "zero":
            baseline_single = torch.zeros(1, num_genes, dtype=torch.float32, device=DEVICE)

        elif baseline_type == "mean_train":
            baseline_single = torch.from_numpy(X_train_mean).float().unsqueeze(0).to(DEVICE)

        elif baseline_type == "mean_shortlist":
            baseline_single = torch.from_numpy(X_dense.mean(axis=0)).float().unsqueeze(0).to(DEVICE)

        elif baseline_type == "random_train":
            # X_train_random has 1000 samples; match to num_samples
            if num_samples <= len(X_train_random):
                baseline_random = torch.from_numpy(X_train_random[:num_samples]).float().to(DEVICE)
            else:
                # If more cells than precomputed samples, sample with replacement
                rng = np.random.default_rng(seed=run_number)
                idxs = rng.choice(len(X_train_random), size=num_samples, replace=True)
                baseline_random = torch.from_numpy(X_train_random[idxs]).float().to(DEVICE)

        elif baseline_type == "random_shortlist":
            rng = np.random.default_rng(seed=run_number)
            random_idxs = rng.choice(num_samples, size=num_samples, replace=True)
            baseline_random = torch.from_numpy(X_dense[random_idxs]).float().to(DEVICE)

        else:
            raise ValueError(f"Unknown baseline_type: {baseline_type}")

        # --- Compute IG in Batches ---
        forward_fn = model_forward(model)
        ig = IntegratedGradients(forward_fn)

        print(f"Computing IG for {num_samples} cells...")
        start_time = time.time()

        ig_values_batches = []
        convergence_deltas_batches = []

        for i in tqdm(range(0, num_samples, BATCH_SIZE), desc="Computing IG"):
            batch_data = X_dense[i: i + BATCH_SIZE]
            batch_tensor = torch.from_numpy(batch_data).float().to(DEVICE)
            batch_tensor.requires_grad_(True)

            # Construct baseline for this batch
            if baseline_type in ("zero", "mean_train", "mean_shortlist"):
                batch_baseline = baseline_single.expand(batch_tensor.shape[0], -1)
            elif baseline_type in ("random_train", "random_shortlist"):
                batch_baseline = baseline_random[i: i + BATCH_SIZE]

            # Compute attributions with convergence delta
            attrs, deltas = ig.attribute(
                batch_tensor,
                baselines=batch_baseline,
                n_steps=N_STEPS,
                method='gausslegendre',
                return_convergence_delta=True,
            )

            ig_values_batches.append(attrs.detach().cpu().numpy())
            convergence_deltas_batches.append(deltas.detach().cpu().numpy())

        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per cell: {total_time / num_samples:.4f} seconds")

        # Concatenate results
        ig_values = np.vstack(ig_values_batches)
        convergence_deltas = np.concatenate(convergence_deltas_batches)

        print(f"Mean convergence delta: {np.mean(np.abs(convergence_deltas)):.6f}")
        print(f"Max convergence delta: {np.max(np.abs(convergence_deltas)):.6f}")

        # Save in same format as DeepSHAP for downstream compatibility
        results = {
            'cell_type': target_cell_type,
            'gene_names': gene_names,
            'ig_values': ig_values,           # analogous to 'shap_values'
            'data_values': X_dense,
            'convergence_deltas': convergence_deltas,
            'baseline_type': baseline_type,
            'n_steps': N_STEPS,
            'total_time_seconds': total_time,
        }

        with open(save_full_path, 'wb') as f:
            pickle.dump(results, f)

        print(f"Saved results to {save_full_path}")

        # Free GPU memory
        del model, ig, forward_fn
        if 'baseline_single' in dir():
            del baseline_single
        if 'baseline_random' in dir():
            del baseline_random
        torch.cuda.empty_cache()

    print("\nIG computation complete.")


if __name__ == "__main__":
    main()

