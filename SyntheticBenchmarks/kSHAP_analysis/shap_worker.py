import sys
import pickle
import numpy as np
import shap
import torch
import time
import os
import compute_accuracy as ca
from model import FeedForwardNet

# ---------------------------------------------------------
# Command Line Arguments
# Expected: python3 shap_worker.py <dataset> <nsamples> <run_number>
# ---------------------------------------------------------
dataset = sys.argv[1]
nsamples = int(sys.argv[2])
run_number = int(sys.argv[3])

# Hardcoded background size to 100 to prevent compute explosion
n_background = 100 

DEVICE = torch.device('cpu')

print(f"--- SHAP Analysis for {dataset} (Run {run_number}) ---")
print(f"Device: {DEVICE}")
print(f"Background samples: {n_background}")
print(f"Coalition samples (nsamples): {nsamples}")

# --- 1. Load Data ---
DATA_FILE = 'synthetic_data.p'
with open(DATA_FILE, 'rb') as f:
    data_dict = pickle.load(f)[dataset]

with open('shortlisted_data.p', 'rb') as f:
    shortlisted_idxs = pickle.load(f)[dataset]

X_all = data_dict['X']
train_idxs = data_dict['train_idxs']

# Background data comes from the training set
X_train = X_all[train_idxs]

# Target data uses the unclipped shortlisted indices
input_samples = X_all[shortlisted_idxs]

# Conditionally load datatypes for the compute_accuracy script
input_datatypes = data_dict['datatypes']
if input_datatypes is not None:
    input_datatypes = input_datatypes[shortlisted_idxs]

n_explain = len(input_samples)

# --- 2. Load Model ---
MODEL_PATH = f"saved_models/best_model_{dataset}.pth"
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    sys.exit(1)

input_dim = X_train.shape[1]
model = FeedForwardNet(input_dim).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# --- 3. Define QOI ---
def QOI(x_numpy):
    x_tensor = torch.from_numpy(x_numpy).float().to(DEVICE)
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.sigmoid(logits)
    return probs.cpu().numpy().flatten()

# --- 4. Run SHAP ---
tic = time.time()

# Summarize background data using K-Means down to exactly 100 centroids
X_background = shap.kmeans(X_train, n_background) 

explainer = shap.KernelExplainer(QOI, X_background)

# Calculate SHAP values utilizing the controlled nsamples budget
print(f"Explaining {n_explain} samples...")
shap_values = explainer.shap_values(input_samples, nsamples=nsamples) 

if isinstance(shap_values, list):
    shap_values = shap_values[0]

toc = time.time()
time_taken = toc - tic
print(f"SHAP calculation done in {time_taken:.2f} seconds.")

# --- 5. Compute Top-K Accuracy ---
# Absolute value ranks strong negative predictors properly
abs_shap_values = np.abs(shap_values)

# acc is an N x k uint8 matrix
acc = ca.compute_correct(abs_shap_values, dataset, input_datatypes)

# --- 6. Save Results ---
os.makedirs('results', exist_ok=True)
save_filename = f'results/topk_acc_shap_{dataset}_ns{nsamples}_run{run_number}.npy'
np.save(save_filename, acc)

print(f"Saved uint8 accuracy matrix of shape {acc.shape} to {save_filename}")
