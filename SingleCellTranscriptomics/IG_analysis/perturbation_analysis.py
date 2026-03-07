"""
Perturbation analysis for Integrated Gradients attributions at delta*.
Uses the same perturbation protocol as SensX (local bounds from SensX delta*).
Per-cell ranking: each cell uses its own IG-based gene ranking.

Usage:
    python3 perturbation_analysis.py <baseline_type> [run_number]

baseline_type: zero, mean_train, mean_shortlist, random_train, random_shortlist
For random_* baselines, provide run_number as 2nd argument.
"""

import numpy as np
import torch
import glob
import os
import re
import sys
import pickle

sys.path.append('../../sensx/')
import sensx

sys.path.append('../model/')
import model_library as ml
import QOI

# --- Command Line Arguments ---
baseline_type = sys.argv[1]
run_number = int(sys.argv[2]) if len(sys.argv) > 2 else 0

# --- Configuration ---
STABILITY_DIR = 'sensx_stability_profiles'
GLOBAL_BOUNDS_DIR = 'sensx_global_bounds'
IG_RESULTS_DIR = f'IG_results_{baseline_type}'
OUTPUT_DIR = f'perturbation_analysis_ig_{baseline_type}'
N_PERTURBATIONS = 1000
EVAL_BATCH_SIZE = 512
SAMPLE_CHUNK_SIZE = 256
TAU_A = 0.1
NUM_K = 55

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda")

# Load global bounds
global_lower = np.load(os.path.join(GLOBAL_BOUNDS_DIR, 'global_lower.npy'))
global_upper = np.load(os.path.join(GLOBAL_BOUNDS_DIR, 'global_upper.npy'))
global_lower_t = torch.from_numpy(global_lower).to(device=device, dtype=torch.float32)
global_upper_t = torch.from_numpy(global_upper).to(device=device, dtype=torch.float32)
global_range_t = global_upper_t - global_lower_t

num_genes = len(global_lower)

# k values (linear, matching SensX perturbation_analysis.py)
k_values = np.unique(np.linspace(1, num_genes, num=NUM_K).astype(int))
print(f"Number of genes: {num_genes}")
print(f"k values ({len(k_values)}): {k_values}")
print(f"Baseline type: {baseline_type}")
if baseline_type.startswith("random"):
    print(f"Run number: {run_number}")


def sanitize_filename(name):
    return re.sub(r'[^\w\-_\. ]', '_', name).replace(' ', '_')


def evaluate_batched(qoi_func, x, batch_size):
    outs = []
    for b in range(0, x.shape[0], batch_size):
        outs.append(qoi_func(x[b:b + batch_size]))
    out = torch.cat(outs, dim=0)
    if out.ndim > 1:
        out = out[:, 0]
    return out


def perturb_and_eval_percell(qoi_func, data_t, local_lower, local_range,
                             ranked_indices_all, k, n_pert, N, mode='top'):
    """
    Per-cell perturbation: each cell uses its own gene ranking.

    ranked_indices_all: (N, num_genes) array of per-cell gene rankings (descending importance)
    mode: 'top' perturbs the first k genes per cell, 'bottom' perturbs the last k genes per cell.
    """
    all_qoi = np.zeros((n_pert, N), dtype=np.float32)

    # Build per-cell feature index masks
    # For top-k: ranked_indices_all[:, :k]
    # For bottom-k: ranked_indices_all[:, -k:]
    if mode == 'top':
        feat_idx_percell = torch.from_numpy(
            ranked_indices_all[:, :k].copy()).long().to(device)  # (N, k)
    else:
        feat_idx_percell = torch.from_numpy(
            ranked_indices_all[:, -k:].copy()).long().to(device)  # (N, k)

    for p in range(n_pert):
        qoi_chunks = []
        for s0 in range(0, N, SAMPLE_CHUNK_SIZE):
            s1 = min(s0 + SAMPLE_CHUNK_SIZE, N)
            n_chunk = s1 - s0
            chunk = data_t[s0:s1].clone()  # (n_chunk, num_genes)

            # Per-cell indices for this chunk
            idx_chunk = feat_idx_percell[s0:s1]  # (n_chunk, k)

            # Gather local bounds for per-cell selected genes
            ll_sel = torch.gather(local_lower[s0:s1], 1, idx_chunk)  # (n_chunk, k)
            lr_sel = torch.gather(local_range[s0:s1], 1, idx_chunk)  # (n_chunk, k)

            # Generate perturbed values
            noise = torch.rand(n_chunk, k, device=device)
            perturbed_vals = ll_sel + noise * lr_sel  # (n_chunk, k)

            # Scatter perturbed values back into chunk
            chunk.scatter_(1, idx_chunk, perturbed_vals)

            with torch.no_grad():
                out = qoi_func(chunk)
                if out.ndim > 1:
                    out = out[:, 0]
            qoi_chunks.append(out.cpu().numpy())
        all_qoi[p] = np.concatenate(qoi_chunks)

    return all_qoi


def run_perturbation_for_celltype(ig_results_path):
    # Load IG results
    with open(ig_results_path, 'rb') as f:
        results = pickle.load(f)

    ctype = results['cell_type']
    ig_values = results['ig_values']  # (N_cells, N_genes)
    safe_ctype_name = sanitize_filename(ctype).replace(' ', '_').replace('/', '_')

    # Output filename
    if baseline_type.startswith("random"):
        out_name = f'perturbation_{safe_ctype_name}_run{run_number}.npz'
    else:
        out_name = f'perturbation_{safe_ctype_name}.npz'
    out_path = os.path.join(OUTPUT_DIR, out_name)

    if os.path.isfile(out_path):
        print(f"Skipping: {out_path} already exists.")
        return

    print(f"\n===== {ctype} =====")

    # Per-cell ranking: rank genes by absolute IG value per cell (descending)
    ranked_indices_all = np.argsort(-np.abs(ig_values), axis=1)  # (N_cells, N_genes)

    # Load stability profile for delta*
    stability_prof_fname = os.path.join(STABILITY_DIR, f'prof_{safe_ctype_name}.npz')
    stability_profile = np.load(stability_prof_fname)
    characteristic_deltas = sensx.find_optimal_delta(stability_profile, TAU_A)
    delta_star = characteristic_deltas.squeeze()

    # Load data
    data_path = f'../high_confidence_samples/{safe_ctype_name}_high_conf.npy'
    data = np.load(data_path)
    data_t = torch.from_numpy(data).to(dtype=torch.float32, device=device)
    N = data_t.shape[0]

    if not isinstance(delta_star, np.ndarray):
        delta_star = np.full(N, float(delta_star))
    if delta_star.ndim == 0:
        delta_star = np.full(N, float(delta_star))
    delta_star_t = torch.from_numpy(delta_star.astype(np.float32)).to(device)

    # Compute local bounds (same as SensX)
    delta_range = delta_star_t.unsqueeze(1) * global_range_t.unsqueeze(0)
    local_lower = torch.max(global_lower_t.unsqueeze(0), data_t - delta_range)
    local_upper = torch.min(global_upper_t.unsqueeze(0), data_t + delta_range)
    local_range = local_upper - local_lower

    # Load model
    model_path = f'../model/saved_models/model_{safe_ctype_name}.pth'
    qoi_func = QOI.qoi_wrapper(model_path, num_genes, device)

    # Baseline QOI
    with torch.no_grad():
        baseline_qoi = evaluate_batched(qoi_func, data_t, EVAL_BATCH_SIZE).cpu().numpy()

    # Results arrays
    topk_median = np.zeros((len(k_values), N), dtype=np.float32)
    topk_q01 = np.zeros((len(k_values), N), dtype=np.float32)
    topk_q99 = np.zeros((len(k_values), N), dtype=np.float32)
    bottomk_median = np.zeros((len(k_values), N), dtype=np.float32)
    bottomk_q01 = np.zeros((len(k_values), N), dtype=np.float32)
    bottomk_q99 = np.zeros((len(k_values), N), dtype=np.float32)

    for ki, k in enumerate(k_values):
        topk_all = perturb_and_eval_percell(
            qoi_func, data_t, local_lower, local_range,
            ranked_indices_all, k, N_PERTURBATIONS, N, mode='top')
        topk_median[ki] = np.median(topk_all, axis=0)
        topk_q01[ki] = np.percentile(topk_all, 1, axis=0)
        topk_q99[ki] = np.percentile(topk_all, 99, axis=0)

        bottomk_all = perturb_and_eval_percell(
            qoi_func, data_t, local_lower, local_range,
            ranked_indices_all, k, N_PERTURBATIONS, N, mode='bottom')
        bottomk_median[ki] = np.median(bottomk_all, axis=0)
        bottomk_q01[ki] = np.percentile(bottomk_all, 1, axis=0)
        bottomk_q99[ki] = np.percentile(bottomk_all, 99, axis=0)

        print(f"  k={k:>5d}: top-k median(mean)={topk_median[ki].mean():.4f}, "
              f"bottom-k median(mean)={bottomk_median[ki].mean():.4f}")

    np.savez(
        out_path,
        k_values=k_values,
        baseline_qoi=baseline_qoi,
        topk_median=topk_median,
        topk_q01=topk_q01,
        topk_q99=topk_q99,
        bottomk_median=bottomk_median,
        bottomk_q01=bottomk_q01,
        bottomk_q99=bottomk_q99,
    )
    print(f"  Saved to {out_path}")


# --- Main loop ---
ig_files = sorted(glob.glob(os.path.join(IG_RESULTS_DIR, '*.pkl')))
print(f"Found {len(ig_files)} IG result files in {IG_RESULTS_DIR}")

for ig_file in ig_files:
    run_perturbation_for_celltype(ig_file)

print("\nDone.")
