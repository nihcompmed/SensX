import numpy as np
import glob
from scipy.stats import rankdata
import scipy
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

dbfile = open('../cell_type_plot_labels.p', 'rb')
CELL_TYPE_LABELS = pickle.load(dbfile)
dbfile.close()

dirr = 'sensitivity'

nw=20

import numpy as np
from scipy.stats import rankdata

def calculate_sequential_convergence(data_matrix):
    """
    Input: numpy array of shape (30, 1000, 1, 27399)
           Dimensions: (batch, cell, redundant, features)
    Returns: convergence_matrix of shape (1000, 29)
    """
    # 1. Shape normalization
    # Squeeze the redundant dimension and move cell to the first axis
    # Resulting shape: (1000, 30, 27399) -> (cells, steps, features)
    M = np.squeeze(data_matrix, axis=2).transpose(1, 0, 2)
    n_cells, n_steps, n_features = M.shape

    print(f"Input Matrix Normalized: Cells={n_cells}, Steps={n_steps}, Features={n_features}")

    # 2. Compute M_sensi (Cumulative Mean)
    # cumsum along the 'steps' axis
    M_cumsum = np.cumsum(M, axis=1)

    # Create divisor array [1, 2, ..., 30] reshaped for broadcasting: (1, 30, 1)
    divisors = np.arange(1, n_steps + 1).reshape(1, n_steps, 1)
    M_sensi = M_cumsum / divisors

    print("M_sensi (Cumulative Mean) computed. Starting ranking...")

    # 3. Compute M_ranks
    # rankdata is vectorized. axis=-1 ranks the 27,399 features for every cell/step.
    # This is the most memory-intensive step.
    M_ranks = rankdata(M_sensi, axis=-1)

    print("M_ranks computed. Calculating adjacent Spearman correlations...")

    # 4. Fast Spearman Calculation (Pearson on Ranks)
    # Spearman rho = Pearson corr of ranks.
    # Since every row is a permutation of 1..N, the mean and variance are constant.

    # Constants for ranks 1..N
    rank_mean = (n_features + 1) / 2.0
    # sum((rank - rank_mean)**2) = n_features * (n_features**2 - 1) / 12
    denom = n_features * (n_features**2 - 1) / 12.0

    # Center the ranks: (1000, 30, 27399)
    centered_ranks = M_ranks - rank_mean

    # Extract adjacent pairs along the 'steps' axis
    # r1: steps 0 to 28 | r2: steps 1 to 29
    r1 = centered_ranks[:, :-1, :] # (1000, 29, 27399)
    r2 = centered_ranks[:, 1:, :]  # (1000, 29, 27399)

    # Covariance numerator: sum of (r1 * r2) across the features axis
    # Resulting shape: (1000, 29)
    numerator = np.sum(r1 * r2, axis=2)

    convergence_matrix = numerator / denom

    print(f"Final Convergence Matrix Shape: {convergence_matrix.shape}")

    return convergence_matrix

all_results = dict()

for ctype in tqdm(CELL_TYPE_LABELS):

    fnames = glob.glob(f'{dirr}/sensx_{ctype}_*')

    all_data = []
    
    for fname in fnames:
    
        data = np.load(fname)

        all_data.append(data)
    
    all_data = np.stack(all_data)

    results = calculate_sequential_convergence(all_data)

    all_results[ctype] = results


print('Saving...')
fname = 'all_convergence_spearman.p'
dbfile = open(fname, 'wb')
pickle.dump(all_results, dbfile)
dbfile.close()



    
