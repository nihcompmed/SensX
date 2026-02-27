import pickle
import torch
import numpy as np
import os
from model import FeedForwardNet

# Configuration
DATA_FILE = 'synthetic_data.p'
OUTPUT_FILE = 'shortlisted_data.p'
MODEL_DIR = '../models/saved_models'
CONFIDENCE_THRESHOLD = 0.99
TARGET_COUNT = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def shortlist_samples():
    print(f"Loading data from {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found. Please run make_data_main.py first.")
        
    with open(DATA_FILE, 'rb') as f:
        data_dict = pickle.load(f)

    shortlisted_results = {}

    for datatype in ['XOR', 'orange_skin', 'nonlinear_additive', 'switch']:
        print(f"\n--- Processing {datatype} ---")
        
        # 1. Initialize Model and Load Weights
        # We determine input dimension from the data shape
        X_all = data_dict[datatype]['X']
        input_dim = X_all.shape[1]
        
        model = FeedForwardNet(input_dim).to(DEVICE)
        model_path = os.path.join(MODEL_DIR, f"best_model_{datatype}.pth")
        
        if not os.path.exists(model_path):
            print(f"Warning: Model file {model_path} not found. Skipping {datatype}.")
            continue
            
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.eval()

        # 2. Get Test Data
        # We use the 'test_idxs' stored in the pickle file (the 20% holdout).
        # Note: train_models.py splits this further into Val/Test, but we use the 
        # full available test pool here to maximize chances of finding 5000 samples.
        test_idxs_original = data_dict[datatype]['test_idxs']
        X_test = X_all[test_idxs_original]
        
        # 3. Run Inference
        X_test_tensor = torch.Tensor(X_test).to(DEVICE)
        
        with torch.no_grad():
            # predict_proba applies sigmoid to logits
            probs = model.predict_proba(X_test_tensor).cpu().numpy().flatten()
            
        # 4. Filter for Label 1 with > 99% Confidence
        # We look for indices relative to the X_test array
        high_conf_mask = probs > CONFIDENCE_THRESHOLD
        candidate_indices_relative = np.where(high_conf_mask)[0]
        
        count_found = len(candidate_indices_relative)
        print(f"  Found {count_found} samples with > {CONFIDENCE_THRESHOLD} confidence (Class 1).")
        
        if count_found == 0:
            print("  No samples matched criteria.")
            shortlisted_results[datatype] = np.array([])
            continue

        # 5. Select up to 5000 samples
        if count_found > TARGET_COUNT:
            print(f"  Subsampling {TARGET_COUNT} from {count_found} candidates...")
            # Randomly choose 5000 without replacement
            selected_relative = np.random.choice(candidate_indices_relative, size=TARGET_COUNT, replace=False)
        else:
            print(f"  Keeping all {count_found} candidates.")
            selected_relative = candidate_indices_relative
            
        # 6. Map back to Original Indices (0 to N-1)
        # We need the indices strictly referring to the original X matrix
        selected_original_idxs = test_idxs_original[selected_relative]
        
        # Store in results
        shortlisted_results[datatype] = selected_original_idxs
        
        # Optional: Validation print
        print(f"  Saved {len(selected_original_idxs)} indices for {datatype}.")

    # 7. Save Results
    print(f"\nSaving shortlisted indices to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(shortlisted_results, f)
    print("Done.")

if __name__ == "__main__":
    shortlist_samples()

