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
    parser = argparse.ArgumentParser(description="Step 1: Tile-Level Stability Scan (All 9 Tiles)")
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--end_idx", type=int, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--time_qoi", action="store_true")
    args = parser.parse_args()

    # --- CONFIGURATION ---
    MODEL_WEIGHTS = 'DeepSpot_pretrained_model_weights/Colon_HEST1K/final_model.pkl'
    MODEL_HPARAM = 'DeepSpot_pretrained_model_weights/Colon_HEST1K/top_param_overall.yaml'
    FEATURE_MODEL_PATH = '/data/aggarwalm4/sensx_compare_morpho_to_spatial/UNI/pytorch_model.bin'
    DATA_FILE = 'predicted_genes_for_sensx.p'
    OUTPUT_IDXS = [1102, 2691, 4673, 4975]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        device=DEVICE, output_idxs=OUTPUT_IDXS, max_batch_size=750 
    )
    qoi_func = TimedQOI(raw_qoi_func, verbose=args.time_qoi)

    # 2. LOAD DATA
    with open(DATA_FILE, 'rb') as f:
        info_dict = pickle.load(f)
    x_input = torch.from_numpy(info_dict['all_XX'][args.start_idx : args.end_idx]).to(DEVICE).float()
    batch_centers = info_dict['all_spot_centers'][args.start_idx : args.end_idx]

    # 3. DEFINE GROUPS (None for Stability)
    # We want independent perturbation of all 270,000 features (9 tiles * 30k features).
    # Passing group_indices=None ensures standard global perturbation.
    
    # 4. STABILITY ANALYSIS
    analyzer = SensitivityAnalyzer(qoi_func, 0, 255, group_indices=None, device=DEVICE)
    
    # Note: Reduced n_s slightly to 500 if memory is tight, but 1000 is standard.
    # We scan stability on the full 9-tile input.
    profile = analyzer.compute_stability_profile(
        x_input, 
        np.linspace(0.1, 1.0, 10), 
        n_s=1000, 
        eval_batch_size=10
    )
    
    delta_star = analyzer.find_optimal_delta(profile, tau_a=0.5)

    # 5. SAVE
    if not os.path.exists(args.out_dir): os.makedirs(args.out_dir)
    save_path = os.path.join(args.out_dir, f"stability_tile_{args.start_idx}_{args.end_idx}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump({
            'delta_star': delta_star.cpu().numpy(),
            'centers': batch_centers,
            'indices': np.arange(args.start_idx, args.end_idx),
            'gene_ids': OUTPUT_IDXS
        }, f)
    print(f"Tile-level stability metadata saved to {save_path}")

if __name__ == "__main__":
    main()
