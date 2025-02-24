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



#do_class = 4
#method_hyperpar = 100
#save_suffix = 0

do_class = int(sys.argv[1])
method_hyperpar = int(sys.argv[2])
save_suffix = sys.argv[3]

#n_trunc = 100

######################################################
# Load model


model_name = f'models/optimal_individual_model_class{do_class}.p'

model = hf.load_model(model_name)


######################################################
# Load test data

fname = f'test_shortlisted/shortlisted_test_class{do_class}.npy'

X_test = np.load(fname)


######################################################

# SENSEX specific

global_domain_low = np.load('global_training_domain_low.npy')
global_domain_high = np.load('global_training_domain_high.npy')


# LOAD QOI
QOI = hf.QOI


# Create sensi object
sensex_obj = sensex.SENSEX(model\
                    , QOI\
                    , global_domain_low\
                    , global_domain_high\
                    )

load_fname = f'delta_star/sensex_analysis_class{do_class}_delta_star.npy'
delta_star = np.load(load_fname)

#X_test = X_test[:n_trunc]
#delta_star = delta_star[:n_trunc]

tic = time.time()

#sensex_vals =\
#        sensex_obj.sensi_batch_features(jnp.array(X_test)\
#                                    , delta_star=jnp.array(delta_star)\
#                                    , nw=method_hyperpar\
#                                    )


sensex_vals =\
        sensex_obj.sensi_batch_inputs(jnp.array(X_test)\
                                    , delta_star=jnp.array(delta_star)\
                                    , nw=method_hyperpar\
                                    )

jax.block_until_ready(1)
toc = time.time()


save_fname = f'batch_inputs_res/sensex_analysis_class{do_class}_batch{save_suffix}_batchinputs.npy'

print(f'Saving results to {save_fname}...')
np.save(save_fname, sensex_vals)
