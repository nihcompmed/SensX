import sys
sys.path.append('../../sensx/') # Uncomment if needed
import sensx
import numpy as np

#############################################################

tau_a = 0.1

for img_name in ['000276', '000375']:

    for model_name in ['Smiling', 'Eyeglasses']:

        stability_prof_fname = f'stability_profiles/prof_{img_name}_{model_name}.npz'
        
        stability_profile = np.load(stability_prof_fname)
        
        characteristic_deltas = sensx.find_optimal_delta(stability_profile, tau_a)
        
        print(img_name, model_name, characteristic_deltas)




