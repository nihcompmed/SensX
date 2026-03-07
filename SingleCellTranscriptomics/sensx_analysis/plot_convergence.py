import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

dbfile = open('../cell_type_plot_labels.p', 'rb')
CELL_TYPE_LABELS = pickle.load(dbfile)
dbfile.close()

def plot_spearman_convergence(pickle_path, output_filename="spearman_convergence.png"):
    """
    Reads a dictionary: {cell_type: (1000, 29)}
    Plots Mean +/- Std Dev for each cell type.
    """
    if not os.path.exists(pickle_path):
        print(f"Error: File {pickle_path} not found.")
        return

    with open(pickle_path, 'rb') as f:
        info = pickle.load(f)

    
    # X-axis represents the 29 adjacent comparisons (1-2, 2-3, ..., 29-30)
    x = np.arange(1, 30)*20 

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    for cell_type, data in info.items():
        # Ensure data is a numpy array for calculations
        data = np.array(data)
        
        # Diagnostics: Check shape integrity
        if data.ndim != 2 or data.shape[1] != 29:
            print(f"Warning: Skipping {cell_type}. Expected shape (N, 29), got {data.shape}")
            continue

        # Calculate statistics across the 1000 cells (axis 0)
        mean_vals = np.mean(data, axis=0)
        std_vals = np.std(data, axis=0)

        axs[0].plot(x, mean_vals, label=f"{CELL_TYPE_LABELS[cell_type]}", lw=2)
        axs[1].plot(x, std_vals, label=f"{CELL_TYPE_LABELS[cell_type]}", lw=2)

        ## Plot the mean line
        #line, = plt.plot(x, mean_vals, label=f"{CELL_TYPE_LABELS[cell_type]}", lw=2)
        #
        ## Plot the shaded standard deviation
        #plt.fill_between(x, 
        #                 mean_vals - std_vals, 
        #                 mean_vals + std_vals, 
        #                 color=line.get_color(), 
        #                 alpha=0.2)

    #plt.title("Feature Ranking Convergence (Spearman Correlation)", fontsize=14)
    axs[0].set_xlabel("Number of walks", fontsize=20)
    axs[0].set_ylabel("Spearman correlation\nmean across 1000 cells", fontsize=20)
    axs[0].set_xticks(x) 
    axs[0].tick_params(axis='x', rotation=45, labelsize=16)
    axs[0].tick_params(axis='y', labelsize=16)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend(loc='lower right', frameon=True, fontsize=12)

    axs[1].set_xlabel("Number of walks", fontsize=20)
    axs[1].set_ylabel("Spearman correlation\nstd.dev. across 1000 cells", fontsize=20)
    axs[1].set_xticks(x) 
    axs[1].tick_params(axis='x', rotation=45, labelsize=16)
    axs[1].tick_params(axis='y', labelsize=16)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend(loc='upper right', frameon=True, fontsize=12)

    #plt.ylim(0.6, 1.05) # Correlations capped at 1.0
    
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")

# To run:
fname = 'all_convergence_spearman.p'
plot_spearman_convergence(fname)
