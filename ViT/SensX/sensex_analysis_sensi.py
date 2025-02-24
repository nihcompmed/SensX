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



dataset = sys.argv[1]
method_hyperpar = int(sys.argv[2])
save_suffix = sys.argv[3]

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


load_fname = f'sensex_analysis_{dataset}_delta_star.npy'
delta_star = np.load(load_fname)


# Create sensi object
sensex_obj = sensex.SENSEX(model\
                    , QOI\
                    , global_hypercube_low\
                    , global_hypercube_high\
                    )

do_samples = 5
X_test = X_test[:do_samples]
delta_star = delta_star[:do_samples]

tic = time.time()

sensex_vals =\
        sensex_obj.sensi_batch_features(jnp.array(X_test)\
                                    , delta_star=jnp.array(delta_star)\
                                    , nw=method_hyperpar\
                                    )

jax.block_until_ready(1)
toc = time.time()


save_fname = f'sensex_analysis_{dataset}_batch{save_suffix}.npy'

print(f'Saving results to {save_fname}...')
np.save(save_fname, sensex_vals)
