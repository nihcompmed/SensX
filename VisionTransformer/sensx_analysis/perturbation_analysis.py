import numpy as np
import torch
import glob
import os
import sys

sys.path.append('../../sensx/')
import sensx
from QOI import initialize_model_and_qoi

# --- Configuration ---
IMG_NAMES = ['000276', '000375']
MODEL_NAMES = ['Smiling', 'Eyeglasses']
N_W = 20
N_PERTURBATIONS = 1000
N_K_VALUES = 50
GLOBAL_LOWER = 0
GLOBAL_UPPER = 1
TAU_A = 0.1
BATCH_SIZE = 100

SENSITIVITY_DIR = 'sensitivity'
STABILITY_DIR = 'stability_profiles'
DATA_DIR = '../model/data'
OUTPUT_DIR = 'perturbation_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_average_sensitivity(img_name, model_name):
    """Load all batch files and average to get final sensitivity map."""
    batch_files = sorted(glob.glob(
        os.path.join(SENSITIVITY_DIR, f'sensx_{img_name}_{model_name}_nw{N_W}_batch*.npy')
    ))
    if len(batch_files) == 0:
        raise FileNotFoundError(f"No batch files found for {img_name}_{model_name}")
    
    arrays = [np.load(f) for f in batch_files]
    stacked = np.stack(arrays, axis=0)
    averaged = np.mean(stacked, axis=0)
    return averaged.squeeze()  # (3, 224, 224)

def get_characteristic_delta(img_name, model_name):
    """Load stability profile and compute characteristic delta."""
    stability_prof_fname = os.path.join(STABILITY_DIR, f'prof_{img_name}_{model_name}.npz')
    stability_profile = np.load(stability_prof_fname)
    characteristic_deltas = sensx.find_optimal_delta(stability_profile, TAU_A)
    return characteristic_deltas.squeeze()  # scalar

def get_k_values(total_features):
    """Generate k values: log-spaced overall + linear fill at the top end."""
    log_ks = np.geomspace(1, total_features, num=N_K_VALUES).astype(int)
    linear_top = np.linspace(100000, total_features, num=20).astype(int)
    k_values = np.unique(np.concatenate([log_ks, linear_top]))
    return k_values

def run_perturbation_experiment(img_name, model_name):
    """Run top-k and bot-k perturbation experiments for one (image, model) case."""
    print(f"\n=== Processing {img_name} / {model_name} ===")
    
    # 1. Load model and QOI
    model_path = f'../model/vit-{model_name}-model-final/'
    qoi_func, transform = initialize_model_and_qoi(model_path, DEVICE)
    
    # 2. Load and preprocess image
    from PIL import Image
    img_path = os.path.join(DATA_DIR, f'{img_name}.jpg')
    raw_image = Image.open(img_path).convert("RGB")
    t_img = transform(raw_image)  # (C, H, W)
    
    # 3. Load sensitivity and get ranking
    sensitivity = load_and_average_sensitivity(img_name, model_name)
    total_features = sensitivity.size  # 150528
    sens_flat = sensitivity.reshape(-1)
    ranking = np.argsort(-sens_flat)  # descending: index 0 = most important
    
    # 4. Get characteristic delta
    delta_star = get_characteristic_delta(img_name, model_name)
    print(f"  Characteristic delta: {delta_star}")
    
    # 5. Compute perturbation bounds
    t_img_flat = t_img.reshape(-1).numpy().astype(np.float64)
    global_range = GLOBAL_UPPER - GLOBAL_LOWER
    delta_range = delta_star * global_range
    local_lower = np.maximum(GLOBAL_LOWER, t_img_flat - delta_range)
    local_upper = np.minimum(GLOBAL_UPPER, t_img_flat + delta_range)
    
    # 6. Get baseline QOI
    with torch.no_grad():
        baseline_qoi = qoi_func(t_img.unsqueeze(0).to(DEVICE)).cpu().numpy().item()
    print(f"  Baseline QOI: {baseline_qoi:.6f}")
    
    # 7. Generate k values
    k_values = get_k_values(total_features)
    print(f"  Number of k values: {len(k_values)}")
    print(f"  k range: {k_values[0]} to {k_values[-1]}")
    
    # 8. Run perturbations
    top_k_results = np.zeros((len(k_values), N_PERTURBATIONS))
    bot_k_results = np.zeros((len(k_values), N_PERTURBATIONS))
    
    for ki, k in enumerate(k_values):
        print(f"  k = {k} ({ki+1}/{len(k_values)})")
        
        top_k_indices = ranking[:k]
        bot_k_indices = ranking[-k:]
        
        # Process in batches
        n_batches = int(np.ceil(N_PERTURBATIONS / BATCH_SIZE))
        
        for b in range(n_batches):
            batch_start = b * BATCH_SIZE
            batch_end = min((b + 1) * BATCH_SIZE, N_PERTURBATIONS)
            current_batch_size = batch_end - batch_start
            
            # --- Top-k perturbation ---
            top_batch = np.tile(t_img_flat, (current_batch_size, 1))
            rand_vals = np.random.uniform(0, 1, size=(current_batch_size, k))
            perturbed_vals = local_lower[top_k_indices] + rand_vals * (local_upper[top_k_indices] - local_lower[top_k_indices])
            top_batch[:, top_k_indices] = perturbed_vals
            
            top_tensor = torch.tensor(top_batch, dtype=torch.float32).reshape(
                current_batch_size, *t_img.shape
            ).to(DEVICE)
            
            with torch.no_grad():
                top_qoi = qoi_func(top_tensor).cpu().numpy().flatten()
            top_k_results[ki, batch_start:batch_end] = top_qoi
            
            # --- Bot-k perturbation ---
            bot_batch = np.tile(t_img_flat, (current_batch_size, 1))
            rand_vals = np.random.uniform(0, 1, size=(current_batch_size, k))
            perturbed_vals = local_lower[bot_k_indices] + rand_vals * (local_upper[bot_k_indices] - local_lower[bot_k_indices])
            bot_batch[:, bot_k_indices] = perturbed_vals
            
            bot_tensor = torch.tensor(bot_batch, dtype=torch.float32).reshape(
                current_batch_size, *t_img.shape
            ).to(DEVICE)
            
            with torch.no_grad():
                bot_qoi = qoi_func(bot_tensor).cpu().numpy().flatten()
            bot_k_results[ki, batch_start:batch_end] = bot_qoi
    
    # 9. Save results
    save_path = os.path.join(OUTPUT_DIR, f'perturbation_{img_name}_{model_name}.npz')
    np.savez(save_path,
             k_values=k_values,
             top_k_qoi=top_k_results,
             bot_k_qoi=bot_k_results,
             baseline_qoi=baseline_qoi,
             delta_star=delta_star)
    print(f"  Saved results to {save_path}")
    print(f"  top_k_qoi shape: {top_k_results.shape}")
    print(f"  bot_k_qoi shape: {bot_k_results.shape}")

def main():
    for img_name in IMG_NAMES:
        for model_name in MODEL_NAMES:
            run_perturbation_experiment(img_name, model_name)
    print("\nDone.")

if __name__ == '__main__':
    main()
