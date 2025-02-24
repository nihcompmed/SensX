import numpy as np
import pickle
import model_library as ml

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random as jrnd
from flax import nnx

import shap

import sys

import random

import time
import compute_accuracy as ca

import helper_functions as hf

#dataset = 'nonlinear_additive'
#n_test = 1000
#method_hyperpar = 100
#run_num = 0

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

######################################################

# Load model

model = hf.load_model(dataset)

######################################################

# Load training data

X_train, X_test, test_datatypes = hf.load_data(dataset, n_test)

######################################################

# SHAP specific

def QOI(x):

    model.eval()

    logits = model(x)

    softmax = jax.nn.softmax(logits, axis=1)

    return softmax[:,1]


tic = time.time()

random_state = random.randint(0,10000)

X_samples = shap.sample(X_train, method_hyperpar, random_state=random_state)

random_state = random.randint(0,10000)

explainer = shap.KernelExplainer(QOI, X_samples)
shap_values = explainer.shap_values(X_test, seed = random_state)

toc = time.time()

all_time_taken = toc - tic

#print(toc - tic)
#print(f'Time taken:{toc - tic}')

# Take absolute of shap because they have sign as direction info
sensi = np.abs(shap_values)

acc = ca.compute_correct(sensi, dataset, test_datatypes)

#print(f'acc:{acc}')

if device == 'cpu':
    save_fname = f'all_SHAP_results_cpu/FACE_{dataset}_ntest{n_test}_nsamples{method_hyperpar}_run{run_num}.p'
elif device == 'gpu':
    save_fname = f'all_SHAP_results_gpu/FACE_{dataset}_ntest{n_test}_nsamples{method_hyperpar}_run{run_num}.p'

res_dict = dict() 
res_dict['dataset'] = dataset
res_dict['ntest'] = n_test
res_dict['method_hyperpar_nsamples'] = method_hyperpar
res_dict['time_taken'] = all_time_taken
res_dict['shap_values'] = shap_values
res_dict['acc'] = acc

dbfile = open(save_fname, 'wb')
pickle.dump(res_dict, dbfile)
dbfile.close()
