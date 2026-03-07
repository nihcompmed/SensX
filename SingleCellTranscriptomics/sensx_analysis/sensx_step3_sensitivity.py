import yaml
import torch
import os
import argparse
import pickle
from PIL import Image
import sys
import numpy as np
import glob
import re
import time

sys.path.append('../../sensx/') # Uncomment if needed
import sensx

sys.path.append('../model/') # Uncomment if needed
import model as ml

import QOI

#############################################################

sensx_batch_num = sys.argv[1]

#############################################################

def sanitize_filename(name):
    return re.sub(r'[^\w\-_\. ]', '_', name).replace(' ', '_')

save_dir = 'sensitivity'

# SensX
global_lower = np.load('global_bounds/global_lower.npy')
global_upper = np.load('global_bounds/global_upper.npy')

# Sensitivity params

tau_a = 0.1

n_w = 20

device = torch.device("cuda")
print(f'device {device}')

target_output_indices = [0]

batch_size = 8192

all_models = glob.glob(f'../model/saved_models/model*.pth')



import gc
import torch

def process_single_model(mm, device, global_lower, global_upper, n_w, batch_size, target_output_indices, tau_a):
    """
    Encapsulates a single model's processing to ensure local variables
    are out of scope after the function returns.
    """
    print(f"Processing: {mm}")

    # 1. Pathing Logic
    ctype = mm.split('/')[-1].split('.')[0][6:]
    safe_ctype_name = sanitize_filename(ctype).replace(' ', '_').replace('/', '_')

    save_fname = f'{save_dir}/sensx_{safe_ctype_name}_nw{n_w}_batch{sensx_batch_num}.npy'
    if os.path.isfile(save_fname):
        return


    # 2. Load Profiles and Deltas
    stability_prof_fname = f'stability_profiles/prof_{safe_ctype_name}.npz'
    stability_profile = np.load(stability_prof_fname)
    characteristic_deltas = sensx.find_optimal_delta(stability_profile, tau_a)
    delta_star = characteristic_deltas.squeeze()

    # 3. Load Data and Move to Device
    data_path = f'../high_confidence_samples/{safe_ctype_name}_high_conf.npy'
    data = np.load(data_path)
    # Cast to float64 here if that's what compute_sensitivity uses
    data_tensor = torch.from_numpy(data).to(dtype=torch.float64, device=device)

    # 4. Initialize Model and Analyzer
    model_path = f'../model/saved_models/model_{ctype}.pth'
    qoi_func = QOI.qoi_wrapper(model_path, data.shape[1], device)

    analyzer = sensx.SensitivityAnalyzer(
        qoi_func=qoi_func,
        global_lower=global_lower,
        global_upper=global_upper,
        device=device
    )

    # 5. Computation
    sensx_res = analyzer.compute_sensitivity(
        data_tensor,
        delta_star,
        n_w,
        batch_size,
        target_output_indices,
        precision='float64'
    )

    # 6. Save and Return
    save_fname = f'{save_dir}/sensx_{safe_ctype_name}_nw{n_w}_batch{sensx_batch_num}.npy'
    np.save(save_fname, sensx_res.detach().cpu().numpy())

    # Explicit return to trigger end-of-scope for locals
    return

# --- Main Execution Loop ---

for mm in all_models:

    tic = time.time()
    process_single_model(mm, device, global_lower, global_upper, n_w, batch_size, target_output_indices, tau_a)
    toc = time.time()

    # Force Garbage Collection for the Python objects
    gc.collect()

    # Force VRAM clearance for the PyTorch allocator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



