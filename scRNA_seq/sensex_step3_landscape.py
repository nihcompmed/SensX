import pickle
import numpy as np
import sys
sys.path.append('../library/')
import sensex
import helper_functions as hf
import model_library as ml

global_domain_low = np.load('global_training_domain_low.npy')
global_domain_high = np.load('global_training_domain_high.npy')


# LOAD QOI
QOI = hf.QOI

import numpy as np
import glob

do_clas = int(sys.argv[1])

all_fnames = glob.glob('batch_inputs_res/sensex_analysis_class*batch*_batchinputs.npy')

info_dict = dict()

for fname in all_fnames:

    data = np.load(fname)

    parser = fname.split('/')[-1].split('.')[0].split('_')

    clas = int(parser[2][5:])

    if clas not in info_dict:
        info_dict[clas] = data
    else:
        
        info_dict[clas] += data


import matplotlib.pyplot as plt
import matplotlib
import math
from tqdm import tqdm

# Get landscape

delta_arr = np.geomspace(0.0001, 1, num=50)

all_info_dict = []


fname = f'test_shortlisted/shortlisted_test_class{do_clas}.npy'

X_test = np.load(fname)

sensex_vals = info_dict[do_clas]

model_name = f'models/optimal_individual_model_class{do_clas}.p'

model = hf.load_model(model_name)

# Create sensi object
sensex_obj = sensex.SENSEX(model\
                    , QOI\
                    , global_domain_low\
                    , global_domain_high\
                    )

for sample, sample_sensex in tqdm(zip(X_test, sensex_vals)):

    
    plt_delta\
    , plt_topn\
    , plt_qoi =\
                sensex_obj.compute_ranks_QOI_landscape(\
                            sample\
                            , sample_sensex\
                            , delta_arr=delta_arr\
                            )
    
    all_info_dict.append([plt_delta, plt_topn, plt_qoi])


import pickle

dbfile = open(f'landscape_info_clas{do_clas}.p', 'wb')
pickle.dump(all_info_dict, dbfile)
dbfile.close()


