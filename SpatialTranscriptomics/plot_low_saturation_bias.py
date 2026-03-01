import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_FILE = 'rigorous_bias_analysis_optimized/saturation_bias_stats.csv'
OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

GENE_ORDER = ['MUC4', 'MUC5B', 'LGALS4', 'TFF3']

def main():
    df = pd.read_csv(RESULTS_FILE)

    # Print exact percentages
    print("=" * 60)
    print("SATURATION BIAS: Percentage with z < -1.96 (p < 0.025)")
    print("=" * 60)
    for gene_name in GENE_ORDER:
        subset = df[df['Gene_Name'] == gene_name]
        n = len(subset)
        n_sig = (subset['Sat_Z'] < -1.96).sum()
        frac_sig = 100 * n_sig / n
        print(f"  {gene_name:10s}: {n_sig}/{n} = {frac_sig:.1f}%")

    n_total = len(df)
    n_sig_total = (df['Sat_Z'] < -1.96).sum()
    frac_sig_total = 100 * n_sig_total / n_total
    print(f"  {'OVERALL':10s}: {n_sig_total}/{n_total} = {frac_sig_total:.1f}%")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, gene_name in enumerate(GENE_ORDER):
        ax = axes[i]
        subset = df[df['Gene_Name'] == gene_name]
        
        ax.hist(subset['Sat_Z'], bins=30, color='teal', edgecolor='black', alpha=0.7)
        ax.axvline(x=-1.96, color='red', linestyle='--', linewidth=2, label='$z = -1.96$')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        median_z = subset['Sat_Z'].median()
        frac_sig = 100 * (subset['Sat_Z'] < -1.96).mean()
        
        ax.text(0.05, 0.95,
                f'Median $z$ = {median_z:.2f}\n{frac_sig:.1f}% with $z < -1.96$',
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f'$\\it{{{gene_name}}}$', fontsize=14)
        ax.tick_params(axis='both', labelsize=11)
        
        if i == 0:
            ax.legend(fontsize=10, loc='upper right')
    
    axes[2].set_xlabel('Z-score (saturation)', fontsize=13)
    axes[3].set_xlabel('Z-score (saturation)', fontsize=13)
    axes[0].set_ylabel('Count (spot-gene pairs)', fontsize=13)
    axes[2].set_ylabel('Count (spot-gene pairs)', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'saturation_bias_zscore_per_gene.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {OUTPUT_DIR}/saturation_bias_zscore_per_gene.png")

if __name__ == '__main__':
    main()
