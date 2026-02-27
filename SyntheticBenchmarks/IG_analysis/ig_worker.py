import sys
import pickle
import numpy as np
import torch
import time
import os

sys.path.append('../utils/')
import compute_accuracy as ca
from model import FeedForwardNet
from captum.attr import IntegratedGradients

# ---------------------------------------------------------
# Command Line Arguments
# Expected: python3 ig_worker.py <dataset> <n_steps> <baseline_type>
#
# baseline_type: "zero", "mean", or "random"
#   - zero:   baseline is the zero vector (deterministic)
#   - mean:   baseline is the mean of training data (deterministic)
#   - random: baseline is a random training sample per input (sampling-based)
#
# For "zero" and "mean", results are deterministic — no repeated runs needed.
# For "random", add a 4th argument: run_number
#   python3 ig_worker.py <dataset> <n_steps> random <run_number>
# ---------------------------------------------------------

dataset = sys.argv[1]
n_steps = int(sys.argv[2])
baseline_type = sys.argv[3]  # "zero", "mean", or "random"
run_number = int(sys.argv[4]) if len(sys.argv) > 4 else 0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"--- Integrated Gradients for {dataset} ---")
print(f"Device: {DEVICE}")
print(f"n_steps: {n_steps}")
print(f"Baseline type: {baseline_type}")
if baseline_type == "random":
    print(f"Run number: {run_number}")

# --- 1. Load Data ---
DATA_FILE = '../data/synthetic_data.p'
with open(DATA_FILE, 'rb') as f:
    data_dict = pickle.load(f)[dataset]

with open('../data/shortlisted_data.p', 'rb') as f:
    shortlisted_idxs = pickle.load(f)[dataset]

X_all = data_dict['X']
train_idxs = data_dict['train_idxs']
X_train = X_all[train_idxs]

input_samples = X_all[shortlisted_idxs]

input_datatypes = data_dict['datatypes']
if input_datatypes is not None:
    input_datatypes = input_datatypes[shortlisted_idxs]

n_explain = len(input_samples)

# --- 2. Load Model ---
MODEL_PATH = f"../models/saved_models/best_model_{dataset}.pth"
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    sys.exit(1)

input_dim = X_train.shape[1]
model = FeedForwardNet(input_dim).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# --- 3. Wrap model for Captum ---
# Captum needs a forward function that returns a tensor.
# For binary classification with a single logit output,
# we attribute the sigmoid probability (matching your QOI).
def model_forward(x):
    logits = model(x)
    return torch.sigmoid(logits).squeeze(-1)

ig = IntegratedGradients(model_forward)

# --- 4. Construct Baseline ---
if baseline_type == "zero":
    # Single zero baseline, broadcast to all inputs
    baseline = torch.zeros(1, input_dim, dtype=torch.float32, device=DEVICE)
elif baseline_type == "mean":
    # Mean of training data
    mean_vec = X_train.mean(axis=0)
    baseline = torch.from_numpy(mean_vec).float().unsqueeze(0).to(DEVICE)
elif baseline_type == "random":
    # One random training sample per input sample
    rng = np.random.default_rng(seed=run_number)
    random_idxs = rng.choice(len(X_train), size=n_explain, replace=True)
    baseline = torch.from_numpy(X_train[random_idxs]).float().to(DEVICE)
else:
    raise ValueError(f"Unknown baseline_type: {baseline_type}. Use 'zero', 'mean', or 'random'.")

# --- 5. Compute IG Attributions ---
input_tensor = torch.from_numpy(input_samples).float().to(DEVICE)
input_tensor.requires_grad_(True)

tic = time.time()
print(f"Computing IG for {n_explain} samples with n_steps={n_steps}...")

# For zero/mean: baseline shape is (1, D), Captum broadcasts automatically.
# For random: baseline shape is (N, D), one per input.
attributions = ig.attribute(
    input_tensor,
    baselines=baseline,
    n_steps=n_steps,
    method='gausslegendre',  # more accurate than riemann
)

ig_values = attributions.detach().cpu().numpy()

toc = time.time()
time_taken = toc - tic
print(f"IG calculation done in {time_taken:.2f} seconds.")

# --- 6. Compute Top-K Accuracy ---
# Absolute value, same as SHAP
abs_ig_values = np.abs(ig_values)

acc = ca.compute_correct(abs_ig_values, dataset, input_datatypes)

# --- 7. Save Results ---
os.makedirs('../results', exist_ok=True)

if baseline_type == "random":
    save_filename = f'../results/topk_acc_ig_{dataset}_ns{n_steps}_{baseline_type}_run{run_number}.npy'
else:
    save_filename = f'../results/topk_acc_ig_{dataset}_ns{n_steps}_{baseline_type}.npy'

np.save(save_filename, acc)
print(f"Saved uint8 accuracy matrix of shape {acc.shape} to {save_filename}")
