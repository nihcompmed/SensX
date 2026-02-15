import numpy as np

def true_top(k, idxs):
    """
    Returns an N x k binary matrix (uint8). 
    1 if the sample is correct at top-ii, 0 otherwise.
    """
    n_samples = idxs.shape[0]
    acc_matrix = np.zeros((n_samples, k), dtype=np.uint8)
    
    for ii in range(k):
        this_idxs = idxs[:, :ii+1]
        maxx = np.amax(this_idxs, axis=1)
        # Check strict correctness per sample
        is_correct = (maxx < k)
        acc_matrix[:, ii] = is_correct.astype(np.uint8)
        
    return acc_matrix

def true_top_v2(true, idxs):
    """
    Returns an N x len(true) binary matrix (uint8) tracking strict top-k accuracy per sample.
    """
    k_len = len(true)
    n_samples = idxs.shape[0]
    acc_matrix = np.zeros((n_samples, k_len), dtype=np.uint8)
    true_arr = np.array(true)

    for ii in range(k_len):
        this_idxs = idxs[:, :ii+1]
        
        # Check if features are in the true set
        is_valid = np.isin(this_idxs, true_arr)
        
        # Strict: All features in the current top-ii+1 must be valid
        is_correct_row = np.all(is_valid, axis=1)
        
        acc_matrix[:, ii] = is_correct_row.astype(np.uint8)
        
    return acc_matrix

def compute_correct(sensi, dataset, datatypes):
    # 2D Check enforcement
    if sensi.ndim != 2:
        raise ValueError(f"sensi must be 2D (samples, features), but got shape {sensi.shape}")

    idxs = np.argsort(-sensi, axis=1)
    total_samples = idxs.shape[0]

    if dataset == 'XOR':
        return true_top(2, idxs)
        
    elif dataset in ['orange_skin', 'nonlinear_additive']:
        true_map = {
            'orange_skin': [0, 1, 2, 3],
            'nonlinear_additive': [0, 1, 2, 3]
        }
        return true_top_v2(true_map[dataset], idxs)

    elif dataset == 'switch':
        # Allocate the final N x 5 matrix to hold results for all samples
        final_matrix = np.zeros((total_samples, 5), dtype=np.uint8)
        
        # Identify subset indices
        orange_mask = (datatypes == 'orange_skin')
        nonlin_mask = (datatypes == 'nonlinear_additive')
        
        orange_idxs = np.argwhere(orange_mask).flatten()
        nonlin_idxs = np.argwhere(nonlin_mask).flatten()

        # Generate N_subset x 5 matrices for each group
        orange_matrix = true_top_v2([0, 1, 2, 3, 9], idxs[orange_idxs])
        nonlin_matrix = true_top_v2([4, 5, 6, 7, 9], idxs[nonlin_idxs])

        # Map the subset matrices back to their original indices in the global matrix
        final_matrix[orange_idxs, :] = orange_matrix
        final_matrix[nonlin_idxs, :] = nonlin_matrix

        return final_matrix

    return None
