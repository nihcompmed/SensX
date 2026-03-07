import sys
sys.path.append('../../sensx/') # Uncomment if needed
import sensx
import numpy as np

#############################################################

tau_a = 0.1

stability_prof_fname = 'stability_profiles/prof_alveolar_macrophage.npz'

stability_profile = np.load(stability_prof_fname)

print(stability_profile.keys())

characteristic_deltas = sensx.find_optimal_delta(stability_profile, tau_a)

print(characteristic_deltas)

