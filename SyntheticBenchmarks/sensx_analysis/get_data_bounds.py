import pickle
import numpy as np
import os

DATA_FILE = '../data/synthetic_data.p'
OUTPUT_DIR = 'global_bounds'

def main():
    print(f"Loading data from {DATA_FILE}...")
    try:
        with open(DATA_FILE, 'rb') as f:
            data_dict = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found.")
        return

    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    print(f"\nComputing feature-wise bounds and saving to '{OUTPUT_DIR}/'...")
    
    for datatype in data_dict.keys():
        X = data_dict[datatype]['X']
        
        # Compute min and max along axis 0 (column-wise / feature-wise)
        # Result shape will be (n_features,) instead of scalar
        feature_min = np.min(X, axis=0)
        feature_max = np.max(X, axis=0)
        
        # Define filenames
        lower_fname = os.path.join(OUTPUT_DIR, f"global_lower_{datatype}.npy")
        upper_fname = os.path.join(OUTPUT_DIR, f"global_upper_{datatype}.npy")
        
        # Save as .npy
        np.save(lower_fname, feature_min)
        np.save(upper_fname, feature_max)
            
        print(f"[{datatype}] Feature count: {len(feature_min)}")
        print(f"    Saved Lower Bounds -> {lower_fname}")
        print(f"    Saved Upper Bounds -> {upper_fname}")

    print("\nDone.")

if __name__ == "__main__":
    main()
