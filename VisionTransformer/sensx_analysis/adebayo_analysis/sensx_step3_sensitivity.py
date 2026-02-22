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

sys.path.append('../../../sensx') 
import sensx


sensx_batch_num = sys.argv[1]

save_dir = 'sensitivity'

#############################################################

# Input to explain
img_name = '000276'

# Model to explain
# Smiling/Eyeglasses
model_name = 'Eyeglasses'

pert_level = 'level_1_block11'
#pert_level = 'level_2_blocks8to11'


#############################################################


# SensX
global_lower = 0
global_upper = 1

# Sensitivity params

tau_a = 0.1

n_w = 20

stability_prof_fname = f'stability_profiles/prof_{img_name}_{model_name}_{pert_level}.npz'
stability_profile = np.load(stability_prof_fname)
characteristic_deltas = sensx.find_optimal_delta(stability_profile, tau_a)

delta_star = characteristic_deltas.squeeze()

print(f'delta star is {delta_star}.')

target_output_indices = [0]

batch_size = 1000

save_fname = f'{save_dir}/sensx_{img_name}_{model_name}_{pert_level}_nw{n_w}_batch{sensx_batch_num}.npy'

#############################################################

device = torch.device("cuda")
print(f'device {device}')

model_path = f'adebayo_{model_name}_cascade/{pert_level}'
qoi_func, transform  = initialize_model_and_qoi(
    model_path,
    device
)

img_path = f'../data/{img_name}.jpg'
raw_image = Image.open(img_path).convert("RGB")
t_img = transform(raw_image) # (C, H, W)

analyzer = sensx.SensitivityAnalyzer(
    qoi_func=qoi_func,
    global_lower=global_lower,
    global_upper=global_upper,
    device=device
)

################
# SENSX EXPECTS [N, *input_shape] where N is the number of 'samples', which is 1 here
################
t_img = torch.unsqueeze(t_img, axis=0)

sensx_res = analyzer.compute_sensitivity(t_img\
                                        , delta_star\
                                        , n_w\
                                        , batch_size\
                                        , target_output_indices\
                                        , precision='float64')


np.save(save_fname, sensx_res.detach().cpu().numpy())


