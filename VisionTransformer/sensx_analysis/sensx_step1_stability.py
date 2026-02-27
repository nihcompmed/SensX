import yaml
import torch
import os
import argparse
import pickle
from PIL import Image
from QOI import initialize_model_and_qoi
import sys
import numpy as np

sys.path.append('../../sensx/')
from sensx import SensitivityAnalyzer


#############################################################

# Input to explain
#img_name = '000276'
img_name = '000375'

# Model to explain
# Smiling/Eyeglasses
model_name = 'Smiling'
#model_name = 'Eyeglasses'

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

model_path = f'../model/vit-{model_name}-model-final/'
qoi_func, transform  = initialize_model_and_qoi(
    model_path,
    device
)

img_path = f'../model/../model/data/{img_name}.jpg'
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


fname = f'{out_dir}/prof_{img_name}_{model_name}.npz'

stability_profile =\
        analyzer.compute_stability_profile(t_img\
                                        , deltas\
                                        , n_s\
                                        , batch_size\
                                        , save_path=fname)



