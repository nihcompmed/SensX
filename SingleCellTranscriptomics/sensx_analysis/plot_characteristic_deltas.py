import glob
import numpy as np
import os
import seaborn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sys
sys.path.append('../../sensx/') # Uncomment if needed
import sensx

import pickle



tau_a = 0.1

out_dir = 'stability_profiles'

fnames = glob.glob(f'{out_dir}/*')



dbfile = open('../cell_type_plot_labels.p', 'rb')
CELL_TYPE_LABELS = pickle.load(dbfile)
dbfile.close()

def get_label(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    key = stem.removeprefix("prof_")
    return CELL_TYPE_LABELS[key]


vals = []
cats = []

for fname in fnames:


    stability_profile = np.load(fname)

    characteristic_deltas = sensx.find_optimal_delta(stability_profile, tau_a)

    vals.append(characteristic_deltas.squeeze())

    cell_label = get_label
    cats.append([get_label(fname)]*characteristic_deltas.shape[0])


vals = np.hstack(vals)
cats = np.hstack(cats)

fig, ax = plt.subplots(figsize=(16, 6))

log_vals = np.log10(vals)

unique_cats = list(dict.fromkeys(cats))

# Sort categories by median delta*
median_order = sorted(unique_cats, key=lambda c: np.median(log_vals[np.array(cats) == c]))
unique_cats = median_order

grouped = [log_vals[np.array(cats) == c] for c in unique_cats]
positions = range(len(unique_cats))

# Alternating background shading
for i in range(len(unique_cats)):
    if i % 2 == 0:
        ax.axvspan(i - 0.5, i + 0.5, color="0.93", zorder=0)

# Violin colors alternate to match background
violin_colors = ["0.70", "0.80"]

parts = ax.violinplot(grouped, positions=positions, showextrema=False)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(violin_colors[i % 2])
    pc.set_edgecolor("0.4")
    pc.set_linewidth(0.5)
    pc.set_alpha(0.7)

for i, g in enumerate(grouped):
    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(g))
    ax.scatter(i + jitter, g, s=12, alpha=0.5, zorder=3)

ax.set_xticks(positions)
ax.set_xticklabels(unique_cats, rotation=45, ha="right", fontsize=14)
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda y, _: f"$10^{{{int(y)}}}$")
)
ax.tick_params(axis="y", labelsize=13)
ax.set_xlabel("")
ax.set_ylabel("Characteristic perturbation factor $\\delta^*$", fontsize=14)
ax.set_xlim(-0.5, len(unique_cats) - 0.5)
plt.tight_layout()

plt.savefig("characteristic_deltas.png", dpi=300, bbox_inches="tight")



