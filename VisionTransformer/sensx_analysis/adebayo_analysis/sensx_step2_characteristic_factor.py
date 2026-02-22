import sys
sys.path.append('../../../sensx/') # Uncomment if needed
import sensx
import numpy as np

#############################################################

img_name = '000276'
model_name = 'Eyeglasses'

tau_a = 0.1

all_pert_levels = [
'level_1_block11'
,'level_2_blocks8to11'
,'level_3_blocks6to11'
,'level_4_blocks0to11'
,'level_5_all']


for pert_level in all_pert_levels:

        stability_prof_fname = f'stability_profiles/prof_{img_name}_{model_name}_{pert_level}.npz'

        stability_profile = np.load(stability_prof_fname)

        print(stability_profile)

        characteristic_deltas = sensx.find_optimal_delta(stability_profile, tau_a)
        
        print(pert_level, characteristic_deltas)




