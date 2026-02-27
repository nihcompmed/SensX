import sys
import QOI
import pickle
import numpy as np
import torch

sys.path.append('../../sensx/')
import sensx

sys.path.append('../utils/')
import compute_accuracy as ca
import os

# ---------------------------------------------------------
# Command Line Arguments
# Expected: python3 sensx_worker.py <model_name> <n_w> <run_number>
# ---------------------------------------------------------
model_name = sys.argv[1]
n_w = int(sys.argv[2])
run_number = int(sys.argv[3])

deltas = np.linspace(0.05, 1, 20) 
tau_a = 0.1

n_s = n_w
batch_size = 100

device = torch.device("cpu")

fname = '../data/synthetic_data.p'
dbfile = open(fname, 'rb')
data = pickle.load(dbfile)[model_name]
dbfile.close()

fname = '../data/shortlisted_data.p'
dbfile = open(fname, 'rb')
shortlisted_idxs = pickle.load(dbfile)[model_name]
dbfile.close()

input_samples = data['X'][shortlisted_idxs]

# CRITICAL FIX: Only slice datatypes if it actually exists in the pickle dict
input_datatypes = data['datatypes']
if input_datatypes is not None:
    input_datatypes = input_datatypes[shortlisted_idxs]

input_dim = input_samples.shape[1]

global_lower = np.load(f'global_bounds/global_lower_{model_name}.npy')
global_upper = np.load(f'global_bounds/global_upper_{model_name}.npy')

model_path = f'../models/saved_models/best_model_{model_name}.pth'

qoi_func = QOI.qoi_wrapper(model_path\
                            , input_dim\
                            , device)

analyzer = sensx.SensitivityAnalyzer(
    qoi_func=qoi_func,
    global_lower=global_lower,
    global_upper=global_upper,
    device=device
)

input_samples = torch.tensor(input_samples)

stability_profile =\
        analyzer.compute_stability_profile(input_samples\
                                        , deltas\
                                        , n_s\
                                        , batch_size\
                                        )

characteristic_deltas = sensx.find_optimal_delta(stability_profile, tau_a)
characteristic_deltas = characteristic_deltas.squeeze()


# Save characteristic deltas
char_deltas_np = characteristic_deltas.cpu().numpy() if torch.is_tensor(characteristic_deltas) else characteristic_deltas
deltas_filename = f'../results/char_deltas_{model_name}_nw{n_w}_run{run_number}.npy'
np.save(deltas_filename, char_deltas_np)
print(f"Saved characteristic deltas of shape {char_deltas_np.shape} to {deltas_filename}")

sensx_res = analyzer.compute_sensitivity(input_samples\
                                , characteristic_deltas\
                                , n_w\
                                , batch_size\
                                , target_output_indices=[0]\
                                , precision='float64')

sensx_res = sensx_res.cpu().numpy()
sensx_res = sensx_res[:, 0, :]

# acc is an N x k uint8 matrix
acc = ca.compute_correct(sensx_res, model_name, input_datatypes)

# Serialization to results directory
os.makedirs('../results', exist_ok=True)
save_filename = f'../results/topk_acc_{model_name}_nw{n_w}_run{run_number}.npy'
np.save(save_filename, acc)
print(f"Saved uint8 accuracy matrix of shape {acc.shape} to {save_filename}")

