import yaml
import torch
import os
import argparse
import pickle
from PIL import Image

import sys

sys.path.append('../')
from QOI import initialize_model_and_qoi
import numpy as np

sys.path.append('../../../sensx/') # Uncomment if needed
from sensx import SensitivityAnalyzer


#############################################################

# Input to explain
#000276/000375
img_name = '000276'

# Model to explain
# Smiling/Eyeglasses
model_name = 'Eyeglasses'

# Perturbation level
#level_0_original

all_pert_levels = [
'level_1_block11'
,'level_2_blocks8to11'
,'level_3_blocks6to11'
,'level_4_blocks0to11'
,'level_5_all']

# SensX
global_lower = 0
global_upper = 1

# 1. Stability profile:
deltas = np.linspace(0.02, 1, num=50, endpoint=True)
n_s = 1000
batch_size = 1000
out_dir = 'stability_profiles'

#############################################################


device = torch.device("cuda")
print(f'device {device}')

for pert_level in all_pert_levels:

    print(f'Doing {pert_level}...')

    model_path = f'adebayo_{model_name}_cascade/{pert_level}'
    qoi_func, transform  = initialize_model_and_qoi(
        model_path,
        device
    )
    
    img_path = f'../data/{img_name}.jpg'
    raw_image = Image.open(img_path).convert("RGB")
    t_img = transform(raw_image) # (C, H, W)
    
    analyzer = SensitivityAnalyzer(
        qoi_func=qoi_func,
        global_lower=global_lower,
        global_upper=global_upper,
        device=device
    )
    
    ################
    # SENSX EXPECTS [N, *input_shape] where N is the number of 'samples', which is 1 here
    ################
    t_img = torch.unsqueeze(t_img, axis=0)
    
    
    fname = f'{out_dir}/prof_{img_name}_{model_name}_{pert_level}.npz'
    
    stability_profile =\
            analyzer.compute_stability_profile(t_img\
                                            , deltas\
                                            , n_s\
                                            , batch_size\
                                            , save_path=fname)
    
    
    
