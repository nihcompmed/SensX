import numpy as np
import torch
import glob
import os
import re
import sys
import pickle

sys.path.append('../model/')
import model as ml

sys.path.append('../sensx_analysis/')
import QOI

# --- Configuration ---
DEEPSHAP_DIR = 'deepSHAP_results'
GLOBAL_BOUNDS_DIR = '../sensx_analysis/global_bounds'
OUTPUT_DIR = 'perturbation_analysis_delta1'
N_PERTURBATIONS = 1000
EVAL_BATCH_SIZE = 512
SAMPLE_CHUNK_SIZE = 256
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

# 55 geom spaced k values
k_values = np.unique(np.geomspace(1, num_genes, num=NUM_K).astype(int))
print(f"Number of genes: {num_genes}")
print(f"k values ({len(k_values)}): {k_values}")


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

    if mode == 'top':
        feat_idx_percell = torch.from_numpy(
            ranked_indices_all[:, :k].copy()).long().to(device)
    else:
        feat_idx_percell = torch.from_numpy(
            ranked_indices_all[:, -k:].copy()).long().to(device)

    for p in range(n_pert):
        qoi_chunks = []
        for s0 in range(0, N, SAMPLE_CHUNK_SIZE):
            s1 = min(s0 + SAMPLE_CHUNK_SIZE, N)
            n_chunk = s1 - s0
            chunk = data_t[s0:s1].clone()

            idx_chunk = feat_idx_percell[s0:s1]

            ll_sel = torch.gather(local_lower[s0:s1], 1, idx_chunk)
            lr_sel = torch.gather(local_range[s0:s1], 1, idx_chunk)

            noise = torch.rand(n_chunk, k, device=device)
            perturbed_vals = ll_sel + noise * lr_sel

            chunk.scatter_(1, idx_chunk, perturbed_vals)

            with torch.no_grad():
                out = qoi_func(chunk)
                if out.ndim > 1:
                    out = out[:, 0]
            qoi_chunks.append(out.cpu().numpy())
        all_qoi[p] = np.concatenate(qoi_chunks)

    return all_qoi


def run_perturbation_for_model(ctype, safe_ctype_name):
    print(f"\n===== {ctype} =====")

    out_path = os.path.join(OUTPUT_DIR, f'perturbation_{safe_ctype_name}.npz')
    if os.path.isfile(out_path):
        print(f"  Skipping: {out_path} already exists.")
        return

    # Load DeepSHAP results
    shap_path = os.path.join(DEEPSHAP_DIR, f'{safe_ctype_name}_deep_benchmark.pkl')
    if not os.path.exists(shap_path):
        print(f"  WARNING: {shap_path} not found, skipping.")
        return

    with open(shap_path, 'rb') as f:
        shap_results = pickle.load(f)

    shap_values = shap_results['shap_values']  # (N, n_genes) or (N, n_genes, 1)

    # Ensure 2D
    if shap_values.ndim != 2:
        shap_values = shap_values.reshape(shap_values.shape[0], -1)

    # Per-cell ranking by absolute SHAP value (descending)
    ranked_indices_all = np.argsort(-np.abs(shap_values), axis=1)  # (N, n_genes)

    # Load data
    data_path = f'../high_confidence_samples/{safe_ctype_name}_high_conf.npy'
    data = np.load(data_path)
    data_t = torch.from_numpy(data).to(dtype=torch.float32, device=device)
    N = data_t.shape[0]

    assert shap_values.shape[0] == N, (
        f"Mismatch: {shap_values.shape[0]} SHAP rows vs {N} cells in data"
    )

    # delta = 1.0: local bounds = global bounds
    local_lower = global_lower_t.unsqueeze(0).expand(N, -1)
    local_upper = global_upper_t.unsqueeze(0).expand(N, -1)
    local_range = global_range_t.unsqueeze(0).expand(N, -1)

    # Load model
    model_path = f'../model/saved_models/model_{ctype}.pth'
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

        print(f"  k={k:>5d}: top-k median(avg)={topk_median[ki].mean():.4f}, "
              f"bottom-k median(avg)={bottomk_median[ki].mean():.4f}")

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
all_models = glob.glob('../model/saved_models/model*.pth')

for mm in all_models:
    ctype = mm.split('/')[-1].split('.')[0][6:]
    safe_ctype_name = sanitize_filename(ctype).replace(' ', '_').replace('/', '_')
    run_perturbation_for_model(ctype, safe_ctype_name)

print("\nDone.")
