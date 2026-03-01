import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. Data Loading
df = pd.read_csv('rigorous_bias_analysis_optimized/optimized_bias_stats.csv')

# 2. Statistical Threshold Calculation
# For p < 0.01 in the lower tail (bias towards edges/smaller distance)
z_crit_edge = stats.norm.ppf(0.01)

# 3. Visualization Construction
plt.figure(figsize=(12, 8))

# Replace boxplot with violinplot
# inner='quartile' shows the internal quartiles within the density plot
sns.violinplot(data=df, x='Gene', y='Z_Score', hue='Gene', palette='Set2', inner='quartile')

# Add reference line for Null expectation (Z=0)
plt.axhline(0, color='black', linestyle='-', linewidth=1.5, label='Null Mean (0)')

# Add reference line for Edge Bias significance threshold (p < 0.01, Z < -2.33)
plt.axhline(z_crit_edge, color='red', linestyle='--', linewidth=2, 
            label=f'Edge Bias Signif. (p < 0.01, Z < {z_crit_edge:.2f})')

# Labeling and Formatting
plt.title('Z-Score Distributions: Testing for Edge Bias (Violin Plots)', fontsize=14)
plt.xlabel('Gene ID', fontsize=12)
plt.ylabel('Z-Score', fontsize=12)

# Cleanup Legend: Show only the reference line labels, hide the gene hue labels
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[-2:], labels[-2:], loc='lower right')

plt.grid(axis='y', linestyle='--', alpha=0.6)

# 4. Output Generation
plt.savefig('gene_zscore_violinplots.png')

# Diagnostic Logs
print(f"Calculated Z-threshold for edge bias (p < 0.01): {z_crit_edge}")
print("Plot saved as gene_zscore_violinplots.png")
