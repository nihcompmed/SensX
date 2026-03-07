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

sys.path.append('../../sensx/') # Uncomment if needed
from sensx import SensitivityAnalyzer

sys.path.append('../model/') # Uncomment if needed
import model as ml

import QOI

#############################################################

def sanitize_filename(name):
    return re.sub(r'[^\w\-_\. ]', '_', name).replace(' ', '_')

# SensX
global_lower = np.load('global_bounds/global_lower.npy')
global_upper = np.load('global_bounds/global_upper.npy')

# 1. Stability profile:
deltas = np.geomspace(1e-4, 1, num=50)
n_s = 1000
batch_size = 64
out_dir = 'stability_profiles'

all_models = glob.glob(f'../model/saved_models/model*.pth')

device = torch.device("cuda")
print(f'device {device}')

for mm in all_models:

    ctype = mm.split('/')[-1].split('.')[0][6:]

    print(f'Doing {ctype}...')

    model_path = f'../model/saved_models/model_{ctype}.pth'
    
    safe_ctype_name = sanitize_filename(ctype).replace(' ', '_').replace('/', '_')
    data_path = f'../high_confidence_samples/{safe_ctype_name}_high_conf.npy'
    
    data = np.load(data_path)
    
    data = torch.from_numpy(data).to(dtype=torch.float32, device=device)
    
    num_genes = data.shape[1]
    
    qoi_func = QOI.qoi_wrapper(model_path, num_genes, device)
    
    analyzer = SensitivityAnalyzer(
        qoi_func=qoi_func,
        global_lower=global_lower,
        global_upper=global_upper,
        device=device
    )
    
    fname = f'{out_dir}/prof_{safe_ctype_name}.npz'
    
    stability_profile =\
            analyzer.compute_stability_profile(data\
                                            , deltas\
                                            , n_s\
                                            , batch_size\
                                            , save_path=fname)
    
    
    
