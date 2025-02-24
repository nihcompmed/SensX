import numpy as np
import glob
import sys
from scipy.stats import rankdata
import pickle
from tqdm import tqdm

import os
import sys
sys.path.append('../library/')
import sensex

fname = 'ct_labels2names.p'
dbfile = open(fname, 'rb')
data = pickle.load(dbfile)
dbfile.close()

do_clas = sys.argv[1]


all_fnames = glob.glob(f'batch_inputs_res/sensex_analysis_class{do_clas}_batch*_batchinputs.npy')

sensex_vals = None

for fname in all_fnames:

    data = np.load(fname)

    if sensex_vals is None:
        sensex_vals = data
    else:
        sensex_vals += data

print('Ranking SensX...')

sensex_ranks_forward = rankdata(-sensex_vals, axis=1)
sensex_ranks_reverse = rankdata(sensex_vals, axis=1)

print('Loading landscape...')

fname = f'landscape_info_clas{do_clas}.p'
dbfile = open(fname, 'rb')
landscape_data = pickle.load(dbfile)
dbfile.close()

significant_sets = []

pareto_optimal_info = []

for idx in tqdm(range(len(landscape_data))):
    
        plt_delta, plt_topn, plt_qoi = landscape_data[idx]
    
        landscape_delta, landscape_topn, landscape_med_qoi = sensex.get_qoi_sensex_pert_landscape(plt_delta, plt_topn, plt_qoi)
    
        n_f = int(np.amax(landscape_topn))
    
        xx = []
        yy = []
    
        n1, n2 = landscape_delta.shape
    
        for ii in range(n1):
            for jj in range(n2):
                if landscape_med_qoi[ii, jj] < 0.5:
                    xx.append(landscape_delta[ii, jj])
                    yy.append(landscape_topn[ii, jj])
    
        pareto_delta, pareto_topn = sensex.get_pareto_optimal(xx, yy, n_f)

        pareto_optimal_info.append([pareto_delta, pareto_topn])
    
        pareto_top_features = np.argwhere(sensex_ranks_forward[idx] <= pareto_topn).flatten()

        pareto_optimal_set = np.zeros(n_f)

        pareto_optimal_set[pareto_top_features] = 1

        significant_sets.append(pareto_optimal_set)

significant_sets = np.vstack(significant_sets)

pareto_optimal_info = np.vstack(pareto_optimal_info)

        
fname = f'sensex_res_class{do_clas}.npz'
np.savez(fname, sensex_vals=sensex_vals, sensex_ranks_forward=sensex_ranks_forward, sensex_ranks_reverse=sensex_ranks_reverse, significant_sets=significant_sets, pareto_optimal_info=pareto_optimal_info)
