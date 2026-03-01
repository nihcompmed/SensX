import os
import sys
import torch
import numpy as np
import pickle
import yaml
import argparse
import time

# --- USER IMPORTS ---
from deepspot.utils.utils_image import get_morphology_model_and_preprocess, compute_mini_tiles
from deepspot.spot import model as actual_model
from deepspot.spot import loss as actual_loss
import qoi_wrapper_VALIDATE as qoi_wrapper

sys.path.append('../')
from sensx_grouped_v4 import SensitivityAnalyzer

class TimedQOI:
    def __init__(self, func, verbose=False):
        self.func = func
        self.verbose = verbose

    def __call__(self, xx):
        if not self.verbose: return self.func(xx)
        start_time = time.perf_counter()
        result = self.func(xx)
        end_time = time.perf_counter()
        print(f"[TIMER] qoi_function: {end_time - start_time:.4f}s | Samples: {xx.shape[0]}")
        return result

def main():
    parser = argparse.ArgumentParser(description="Step 2: Tile-Level Grouped Sensitivity Worker")
    parser.add_argument("--task_file", type=str, required=True, help="Path to tile_sensitivity_tasks.p")
    parser.add_argument("--task_start", type=int, required=True)
    parser.add_argument("--task_end", type=int, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--time_qoi", action="store_true")
    args = parser.parse_args()

    # --- CONFIGURATION ---
    MODEL_WEIGHTS = 'DeepSpot_pretrained_model_weights/Colon_HEST1K/final_model.pkl'
    MODEL_HPARAM = 'DeepSpot_pretrained_model_weights/Colon_HEST1K/top_param_overall.yaml'
    FEATURE_MODEL_PATH = '/data/aggarwalm4/sensx_compare_morpho_to_spatial/UNI/pytorch_model.bin'
    DATA_FILE = 'predicted_genes_for_sensx.p'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GLOBAL_GENE_LIST = [1102, 2691, 4673, 4975] 

    # 1. SETUP MODEL & QOI
    sys.modules['deepspot.model'] = actual_model
    sys.modules['deepspot.spot'] = actual_model
    sys.modules['deepspot.loss'] = actual_loss
    
    model_expression = torch.load(MODEL_WEIGHTS, map_location=DEVICE).eval()
    with open(MODEL_HPARAM, "r") as f:
        config = yaml.safe_load(f)
        
    morphology_model, preprocess, _ = get_morphology_model_and_preprocess(
        model_name=config['image_feature_model'],
        device=DEVICE, model_path=FEATURE_MODEL_PATH
    )
    morphology_model.to(DEVICE).eval()
    
    raw_qoi_func = qoi_wrapper.qoi_wrapper(
        resize_shape=(9, 100, 100, 3), model_expression=model_expression, 
        preprocess=preprocess, compute_mini_tiles=compute_mini_tiles,
        detach_and_convert=lambda d: d[None, ].detach().float(), morphology_model=morphology_model, 
        mean=torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1),
        std=torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1),
        device=DEVICE, output_idxs=GLOBAL_GENE_LIST, max_batch_size=250 
    )
    qoi_func = TimedQOI(raw_qoi_func, verbose=args.time_qoi)

    # 2. LOAD TASKS AND DATA
    with open(args.task_file, 'rb') as f:
        all_tasks = pickle.load(f)
    my_tasks = all_tasks[args.task_start : args.task_end]
    
    with open(DATA_FILE, 'rb') as f:
        info_dict = pickle.load(f)
    
    # 3. DEFINE 9 TILE GROUPS
    # The input tensor (9, 100, 100, 3) is flattened to 270,000 features.
    # Tile i corresponds to indices [i*30000 : (i+1)*30000]
    num_tiles = 9
    features_per_tile = 30000
    group_indices = []
    
    for i in range(num_tiles):
        start = i * features_per_tile
        end = (i + 1) * features_per_tile
        # Create tensor on device to avoid overhead during loop
        group_indices.append(torch.arange(start, end, device=DEVICE, dtype=torch.long))

    analyzer = SensitivityAnalyzer(qoi_func, 0, 255, group_indices, DEVICE)
    
    if not os.path.exists(args.out_dir): os.makedirs(args.out_dir)

    # 4. SENSITIVITY COMPUTATION
    for i, task in enumerate(my_tasks):
        spot_idx = task['spot_idx']
        delta_star = task['delta_star']
        target_indices = task['target_gene_indices']
        
        print(f"Task {args.task_start + i}: Spot {spot_idx}, Genes {task['gene_ids']}, Delta {delta_star}")
        
        x_input = torch.from_numpy(info_dict['all_XX'][spot_idx : spot_idx+1]).to(DEVICE).float()
        
        # Update QOI to target only specific genes in the task
        analyzer.qoi_func.func.keywords['output_idxs'] = [GLOBAL_GENE_LIST[gi] for gi in target_indices]

        # CORE COMPUTE: Tile-Level Sensitivity
        # This will return a tensor of shape (1_sample, num_genes, 9_groups)
        final_sensitivity = analyzer.compute_sensitivity(
            x=x_input, 
            delta_star=delta_star, 
            n_w=200, 
            batch_size=128, 
            target_output_indices=list(range(len(target_indices)))
        )
        
        # 5. SAVE
        # We save the raw 9-value vector per gene.
        # final_sensitivity shape: (1, num_genes, 9)
        tile_sens_vectors = final_sensitivity[0].cpu().numpy() # shape (num_genes, 9)

        save_path = os.path.join(args.out_dir, f"tile_sens_spot{spot_idx}_delta{delta_star:.3f}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump({
                'tile_sensitivity': tile_sens_vectors, 
                'delta_star': delta_star,
                'spot_idx': spot_idx,
                'center': task['center'],
                'gene_ids': task['gene_ids']
            }, f)

    print(f"Worker finished tasks {args.task_start} to {args.task_end}")

if __name__ == "__main__":
    main()
