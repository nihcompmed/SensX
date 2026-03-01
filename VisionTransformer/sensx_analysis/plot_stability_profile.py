import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../../sensx/') # Uncomment if needed
import sensx

fname = 'stability_profiles/prof_000276_Smiling.npy'

#############################################################

# Input to explain
img_name = '000276'

# Model to explain
# Smiling/Eyeglasses
model_name = 'Smiling'

tau_a = 0.1

stability_prof_fname = f'stability_profiles/prof_{img_name}_{model_name}.npz'

save_path = f'plots/stability_profile_{img_name}_{model_name}.jpg'

#############################################################

stability_profile = np.load(stability_prof_fname)

#############################################################

characteristic_deltas = sensx.find_optimal_delta(stability_profile, tau_a)

#############################################################



deltas = np.asarray(stability_profile["deltas"])
median = np.asarray(stability_profile["median"])
q1 = np.asarray(stability_profile["q1"])
q99 = np.asarray(stability_profile["q99"])

sample_idx = 0

med = median[:, sample_idx]
lo = q1[:, sample_idx]
hi = q99[:, sample_idx]
label_suffix = ""

fig, ax = plt.subplots(figsize=(12, 8))
ax.fill_between(deltas, lo, hi, alpha=0.25, color="steelblue", label="q1–q99")
ax.plot(deltas, med, color="darkblue", linewidth=1.5, label="median")
ax.set_xlabel("delta")
ax.set_ylabel("QoI")
ax.legend()

title = f"Stability Profile — sample {sample_idx}{label_suffix}"
ax.set_title(title)

ax.axvline(characteristic_deltas.squeeze(), ls='--')

plt.tight_layout()

fig.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"  [Saved] Plot -> {save_path}")



