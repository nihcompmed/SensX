"""
Changes from Previous Version:
- Integrated Precision Sync: Automatically detects and casts inputs to the morphology_model's dtype.
- Added tqdm progress bars for granular sample-level and batch-level tracking.
- Implemented batch-size safety (default 100) to prevent CUDA OOM during UNI forward passes.
- Added explicit .float() conversion for baseline and perturbed results to ensure stable delta calculations.
"""

import os
import sys
import torch
import numpy as np
import pickle
import yaml
import argparse
from tqdm import tqdm

# --- USER IMPORTS ---
from deepspot.utils.utils_image import get_morphology_model_and_preprocess, compute_mini_tiles
from deepspot.spot import model as actual_model
from deepspot.spot import loss as actual_loss
import qoi_wrapper_VALIDATE as qoi_wrapper

def main():
    parser = argparse.ArgumentParser(description="Step 3: Center vs Neighbor Perturbation Test (Ablation Study)")
    parser.add_argument("--task_file", type=str, required=True, help="Path to tile_sensitivity_tasks.p")
    parser.add_argument("--task_start", type=int, required=True)
    parser.add_argument("--task_end", type=int, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=100, help="VRAM safety: 100-250 recommended for UNI")
    args = parser.parse_args()

    # --- CONFIGURATION ---
    MODEL_WEIGHTS = 'DeepSpot_pretrained_model_weights/Colon_HEST1K/final_model.pkl'
    MODEL_HPARAM = 'DeepSpot_pretrained_model_weights/Colon_HEST1K/top_param_overall.yaml'
    FEATURE_MODEL_PATH = '/data/aggarwalm4/sensx_compare_morpho_to_spatial/UNI/pytorch_model.bin'
    DATA_FILE = 'predicted_genes_for_sensx.p'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GLOBAL_GENE_LIST = [1102, 2691, 4673, 4975] 
    N_SAMPLES = 1000

    # 1. SETUP MODEL & QOI
    # Ensure unpickler finds classes in the correct namespace
    sys.modules['deepspot.model'] = actual_model
    sys.modules['deepspot.spot'] = actual_model
    sys.modules['deepspot.loss'] = actual_loss
    
    print(f"Loading model weights from {MODEL_WEIGHTS}...")
    model_expression = torch.load(MODEL_WEIGHTS, map_location=DEVICE).eval()
    
    with open(MODEL_HPARAM, "r") as f:
        config = yaml.safe_load(f)
        
    print(f"Initializing morphology model ({config['image_feature_model']})...")
    morphology_model, preprocess, _ = get_morphology_model_and_preprocess(
        model_name=config['image_feature_model'], device=DEVICE, model_path=FEATURE_MODEL_PATH
    )
    morphology_model.to(DEVICE).eval()
    
    # CRITICAL: Detect required precision (e.g., torch.float16 or torch.float32)
    TARGET_DTYPE = next(morphology_model.parameters()).dtype
    print(f"Target Precision detected: {TARGET_DTYPE}")

    raw_qoi_func = qoi_wrapper.qoi_wrapper(
        resize_shape=(9, 100, 100, 3), model_expression=model_expression, 
        preprocess=preprocess, compute_mini_tiles=compute_mini_tiles,
        detach_and_convert=lambda d: d[None, ].detach().float(), morphology_model=morphology_model, 
        mean=torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1),
        std=torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1),
        device=DEVICE, output_idxs=GLOBAL_GENE_LIST, max_batch_size=args.batch_size 
    )

    # 2. LOAD TASKS AND DATA
    with open(args.task_file, 'rb') as f:
        all_tasks = pickle.load(f)
    my_tasks = all_tasks[args.task_start : args.task_end]
    
    with open(DATA_FILE, 'rb') as f:
        info_dict = pickle.load(f)
    
    if not os.path.exists(args.out_dir): os.makedirs(args.out_dir)

    # 3. PERTURBATION LOOP
    for i, task in enumerate(my_tasks):
        spot_idx = task['spot_idx']
        delta_star = task['delta_star']
        target_indices = task['target_gene_indices']
        
        print(f"\n[Task {args.task_start + i}] Spot: {spot_idx} | Genes: {task['gene_ids']} | Delta*: {delta_star}")
        
        # Load original input and baseline prediction
        x_orig = torch.from_numpy(info_dict['all_XX'][spot_idx : spot_idx+1]).to(DEVICE).float()
        raw_qoi_func.keywords['output_idxs'] = [GLOBAL_GENE_LIST[gi] for gi in target_indices]
        
        with torch.no_grad():
            # Ensure baseline is calculated at target precision but stored as float
            y_baseline = raw_qoi_func(x_orig.to(TARGET_DTYPE)).float()

        # Define bounds (assuming [0, 255] range)
        delta_range = delta_star * 255.0
        lower_bound = torch.clamp(x_orig - delta_range, 0, 255)
        upper_bound = torch.clamp(x_orig + delta_range, 0, 255)

        results = {
            'center_deltas': [],
            'neighbor_deltas': [],
            'gene_ids': task['gene_ids'],
            'spot_idx': spot_idx,
            'delta_star': delta_star
        }

        # --- ABLATION TEST 1: PERTURB CENTER (Tiles 0), FIX NEIGHBORS (Tiles 1-8) ---
        pbar_c = tqdm(total=N_SAMPLES, desc="  Center Perturb ", unit="samples", leave=False)
        for _ in range(N_SAMPLES // args.batch_size):
            x_batch = x_orig.repeat(args.batch_size, 1)
            # Center is first 30k features
            noise = torch.rand((args.batch_size, 30000), device=DEVICE, dtype=torch.float32)
            x_batch[:, :30000] = lower_bound[:, :30000] + noise * (upper_bound[:, :30000] - lower_bound[:, :30000])
            
            with torch.no_grad():
                # Cast to model precision (e.g., float16) for forward pass
                y_perturbed = raw_qoi_func(x_batch.to(TARGET_DTYPE)).float()
                results['center_deltas'].append((y_perturbed - y_baseline).cpu().numpy())
            pbar_c.update(args.batch_size)
        pbar_c.close()

        # --- ABLATION TEST 2: PERTURB NEIGHBORS (Tiles 1-8), FIX CENTER (Tile 0) ---
        pbar_n = tqdm(total=N_SAMPLES, desc="  Neighbor Perturb", unit="samples", leave=False)
        for _ in range(N_SAMPLES // args.batch_size):
            x_batch = x_orig.repeat(args.batch_size, 1)
            # Neighbors are features 30k through 270k
            noise = torch.rand((args.batch_size, 240000), device=DEVICE, dtype=torch.float32)
            x_batch[:, 30000:] = lower_bound[:, 30000:] + noise * (upper_bound[:, 30000:] - lower_bound[:, 30000:])
            
            with torch.no_grad():
                y_perturbed = raw_qoi_func(x_batch.to(TARGET_DTYPE)).float()
                results['neighbor_deltas'].append((y_perturbed - y_baseline).cpu().numpy())
            pbar_n.update(args.batch_size)
        pbar_n.close()

        # Stack distributions for saving
        results['center_deltas'] = np.vstack(results['center_deltas']) # (1000, num_target_genes)
        results['neighbor_deltas'] = np.vstack(results['neighbor_deltas']) # (1000, num_target_genes)

        save_path = os.path.join(args.out_dir, f"perturb_spot{spot_idx}_delta{delta_star:.3f}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)

    print(f"\nWorker finished tasks {args.task_start} to {args.task_end}")

if __name__ == "__main__":
    main()
