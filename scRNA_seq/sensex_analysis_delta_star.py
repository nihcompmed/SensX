import sys
import numpy as np
import pickle
import model_library as ml

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random as jrnd
from flax import nnx

import sys
sys.path.append('../library/')
import sensex

import time

import helper_functions as hf

import glob
from tqdm import tqdm

all_fnames = glob.glob('test_shortlisted/shortlisted_test_class*.npy')

global_domain_low = np.load('global_training_domain_low.npy')
global_domain_high = np.load('global_training_domain_high.npy')

nn_global = 1000

delta_arr = jnp.geomspace(0.0001, 1, num=50)

for fname in all_fnames:

    do_class = int(fname.split('/')[-1].split('.')[0].split('_')[-1][5:])
    
    print(f'Doing class {do_class}...')
    
    model_name = f'models/optimal_individual_model_class{do_class}.p'
    
    model = hf.load_model(model_name)

    fname = f'test_shortlisted/shortlisted_test_class{do_class}.npy'

    X_test = np.load(fname)
    
    # LOAD QOI
    QOI = hf.QOI
    
    # 1. Create sensi object
    sensex_obj = sensex.SENSEX(model\
                        , QOI\
                        , global_domain_low\
                        , global_domain_high\
                        )
    
    
    tic = time.time()
    
    all_qoi, delta_arr_explore_perts = sensex_obj.get_perturbed_qoi_all_info(X_test\
                                                                            , nn_global=nn_global\
                                                                            , delta_arr=delta_arr)
    
    
    delta_star = sensex_obj.get_delta_star(all_qoi, delta_arr_explore_perts)
    
    
    jax.block_until_ready(1)
    toc = time.time()
    
    time_taken = toc - tic
    print(f'Time taken:{time_taken}')
    
    save_fname = f'sensex_analysis_class{do_class}_step1.npz'
    print(f'Saving results to {save_fname}...')
    np.savez(save_fname, all_qoi=all_qoi, delta_arr_explore_perts=delta_arr_explore_perts, time_taken=np.array(time_taken))

    save_fname = f'sensex_analysis_class{do_class}_delta_star.npy'
    print(f'Saving delta_star to {save_fname}...')
    np.save(save_fname, delta_star)








