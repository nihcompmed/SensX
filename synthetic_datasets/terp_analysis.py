import sys
import subprocess
import numpy as np

import pickle

from qoi import QOI

import time
import compute_accuracy as ca

import helper_functions as hf
import jax
jax.config.update("jax_enable_x64", True)



#dataset = 'nonlinear_additive'
#n_test = 10
#method_hyperpar = 100
#run_num = 0

dataset = sys.argv[1]
n_test = int(sys.argv[2])
method_hyperpar = int(sys.argv[3])
run_num = sys.argv[4]


######################################################

# Load model

model = hf.load_model(dataset)

######################################################

# Load data

X_train, X_test, test_datatypes = hf.load_data(dataset, n_test)


######################################################

# TERP specific

terp_dirr = '../TERP'

terp_neigh_script = f'{terp_dirr}/TERP_neighborhood_generator.py'

terp_optim1_script = f'{terp_dirr}/TERP_optimizer_01.py'

terp_optim2_script = f'{terp_dirr}/TERP_optimizer_02.py'

def QOI(x):

    model.eval()

    logits = model(x)

    softmax = jax.nn.softmax(logits, axis=1)

    return softmax[:,1]

# Save X_test as numpy matrix to be used by TERP
input_fname = f'X_test_TERP.npy'
np.save(input_fname, X_test)

def single_input(input_index, input_fname):

    time_taken = 0

    tic = time.time()

    # 1. generate neighborhood samples
    cmd = ['python3', terp_neigh_script, '-seed', '0', '-input_numeric', input_fname, '-num_samples'
    , f'{method_hyperpar}', '-index', f'{input_index}']

    subprocess.run(cmd)

    # 2. predictions of the samples
    pred_1 = QOI(np.load('DATA/make_prediction_numeric.npy'))
    np.save('DATA/neighborhood_state_probabilities.npy', pred_1)

    toc = time.time()

    time_taken += toc-tic

    # NOTE: Do not time this for fair comparison
    # NOTE: Pick cutoff as the total number of features
    # 3. Optim1 for dimensionality reduction

    cmd = ['python3', terp_optim1_script, '-cutoff', '10', '-TERP_input', 'DATA/TERP_numeric.npy', '-blackbox_prediction', 'DATA/neighborhood_state_probabilities.npy']
    subprocess.run(cmd)

    # No need to resample because using full dimensions

    tic = time.time()

    # 4. TERP forward feature selection using the generated neighborhood samples
    cmd = ['python3', terp_optim2_script, '-TERP_input', 'DATA/TERP_numeric.npy', '-blackbox_prediction', 'DATA/neighborhood_state_probabilities.npy', '-selected_features', 'TERP_results/selected_features.npy']
    subprocess.run(cmd)

    toc = time.time()

    time_taken += toc-tic

    w = np.load('TERP_results_2/optimal_feature_weights.npy')

    return w, time_taken

all_res = np.zeros((n_test, 10))

all_time_taken = 0

for ii in range(n_test):

    res, time_taken = single_input(ii, input_fname)

    all_time_taken += time_taken

    all_res[ii] = res



acc = ca.compute_correct(all_res, dataset, test_datatypes)

save_fname = f'all_TERP_results/TERP_{dataset}_ntest{n_test}_nsamples{method_hyperpar}_run{run_num}.p'

res_dict = dict() 
res_dict['dataset'] = dataset
res_dict['ntest'] = n_test
res_dict['method_hyperpar_nsamples'] = method_hyperpar
res_dict['time_taken'] = all_time_taken
res_dict['terp_values'] = all_res
res_dict['acc'] = acc

dbfile = open(save_fname, 'wb')
pickle.dump(res_dict, dbfile)
dbfile.close()





