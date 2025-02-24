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


nn_global = 1000

for dataset in ['Smiling', 'Eyeglasses']:

    print(f'Doing {dataset}...')

    ######################################################
    # Load model
    
    model = hf.load_model(dataset)
    
    
    ######################################################
    # Load test data
    
    X_test = hf.load_shortlisted_data_similing_eyeglasses()
    
    ######################################################
    
    # SENSEX specific
    
    global_hypercube_low, global_hypercube_high  = hf.load_training_hypercube(dataset)
    
    # LOAD QOI
    QOI = hf.QOI
    
    # 1. Create sensi object
    sensex_obj = sensex.SENSEX(model\
                        , QOI\
                        , global_hypercube_low\
                        , global_hypercube_high\
                        )
    
    
    tic = time.time()
    
    all_qoi, delta_arr_explore_perts = sensex_obj.get_perturbed_qoi_all_info(X_test\
                                                                            , nn_global=nn_global)
    
    
    delta_star = sensex_obj.get_delta_star(all_qoi, delta_arr_explore_perts)
    
    
    jax.block_until_ready(1)
    toc = time.time()
    
    time_taken = toc - tic
    print(f'Time taken:{time_taken}')
    
    save_fname = f'sensex_analysis_{dataset}_step1.npz'
    print(f'Saving results to {save_fname}...')
    np.savez(save_fname, all_qoi=all_qoi, delta_arr_explore_perts=delta_arr_explore_perts, time_taken=np.array(time_taken))

    save_fname = f'sensex_analysis_{dataset}_delta_star.npy'
    print(f'Saving delta_star to {save_fname}...')
    np.save(save_fname, delta_star)








