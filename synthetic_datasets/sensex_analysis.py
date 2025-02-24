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
import compute_accuracy as ca

import helper_functions as hf


#
#dataset = 'nonlinear_additive'
#n_test = 1000
#method_hyperpar = 10000
#run_num = 0
#device = 'gpu'

dataset = sys.argv[1]
n_test = int(sys.argv[2])
method_hyperpar = int(sys.argv[3])
run_num = sys.argv[4]
device = sys.argv[5]

if device == 'cpu':
    print('FORCING DEVICE CPU FOR JAX.')
    jax.config.update('jax_default_device', jax.devices('cpu')[0])
elif device == 'gpu':
    print('DOING ON GPU IF FOUND.')
else:
    print('Invalid device argument.')
    exit()


print(f'JAX device {jax.devices()}')

print(f'{dataset} n_test:{n_test} hyperpar:{method_hyperpar}')

######################################################
# Load model

model = hf.load_model(dataset)


######################################################
# Load data

X_train, X_test, test_datatypes = hf.load_data(dataset, n_test)

######################################################

# FACE specific

# LOAD QOI
QOI = hf.QOI

# Get global hypercube
global_hypercube_low = np.amin(X_train, axis=0)
global_hypercube_high = np.amax(X_train, axis=0)

# 1. Create sensi object
sensex_obj = sensex.SENSEX(model\
                    , QOI\
                    , global_hypercube_low\
                    , global_hypercube_high\
                    )

# 2. explore perts

tic = time.time()

all_qoi, delta_arr_explore_perts = sensex_obj.get_perturbed_qoi_all_info(X_test\
                                                                        , nn_global=2000)

jax.block_until_ready(1)
toc = time.time()
time_taken1 = toc - tic
print(f'Time taken to explore perts:{time_taken1}')


tic = time.time()

delta_star = sensex_obj.get_delta_star(all_qoi, delta_arr_explore_perts)

jax.block_until_ready(1)
toc = time.time()
time_taken2 = toc - tic
print(f'Time taken to get delta_star perts:{time_taken2}')

tic = time.time()

sensex_vals =\
        sensex_obj.sensi_batch_inputs(jnp.array(X_test)\
                                    , delta_star=jnp.array(delta_star)\
                                    , nw=method_hyperpar)

jax.block_until_ready(1)
toc = time.time()

time_taken3 = toc - tic
print(f'Time taken to compute sensi:{time_taken3}')

acc = ca.compute_correct(sensex_vals, dataset, test_datatypes)

print(f'acc:{acc}')


res_dict = dict() 
res_dict['dataset'] = dataset
res_dict['ntest'] = n_test
res_dict['method_hyperpar_nsamples'] = method_hyperpar

res_dict['time_taken_explore_perts'] = time_taken1
res_dict['time_taken_compute_delta_star'] = time_taken2
res_dict['time_taken_compute_sensi'] = time_taken3

res_dict['delta_arr_explore_perts'] = delta_arr_explore_perts
res_dict['all_qoi'] = all_qoi
res_dict['delta_star'] = delta_star

res_dict['sensex_vals'] = sensex_vals
res_dict['final_acc'] = acc

if device == 'cpu':
    save_fname = f'all_SENSEX_results_cpu/SENSEX_{dataset}_ntest{n_test}_nsamples{method_hyperpar}_run{run_num}.p'
elif device == 'gpu':
    save_fname = f'all_SENSEX_results_gpu/SENSEX_{dataset}_ntest{n_test}_nsamples{method_hyperpar}_run{run_num}.p'

dbfile = open(save_fname, 'wb')
pickle.dump(res_dict, dbfile)
dbfile.close()




