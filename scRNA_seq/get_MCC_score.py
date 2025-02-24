import sys
import numpy as np
import pickle
import model_library as ml

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random as jrnd
from flax import nnx

import sys
sys.path.append('../library/')
import sensex

import time

import helper_functions as hf

import glob
import math


test_dirr = 'test_shortlisted'

test_fnames = glob.glob(f'{test_dirr}/*')

all_classes = []

for fname in test_fnames:

    parser = fname.split('/')[-1].split('.')[0].split('_')

    clas = parser[2][5:]

    all_classes.append(int(clas))

fname = 'ct_labels2names.p'
dbfile = open(fname, 'rb')
label_names = pickle.load(dbfile)
dbfile.close()

info_file = open('MCC_test_shortlisted.csv', 'w')

info_dict = dict()

for do_class in all_classes:

    model_name = f'models/optimal_individual_model_class{do_class}.p'
    
    model = hf.load_model(model_name)

    tp = 0
    tn = 0

    fp = 0
    fn = 0

    for test_clas in all_classes:

        fname = f'{test_dirr}/shortlisted_test_class{test_clas}.npy'

        XX = np.load(fname)

        XX_qoi = hf.QOI(XX, model)

        ones = np.argwhere(XX_qoi > 0.5).flatten()

        n_ones = len(ones)
        n_zeros = len(XX_qoi) - n_ones

        if test_clas == do_class:

            tp += n_ones
            fn += n_zeros

        else:

            fp += n_ones
            tn += n_zeros




    info_dict[do_class] = dict()

    info_dict['tp'] = tp
    info_dict['tn'] = tn

    info_dict['fp'] = fp
    info_dict['fn'] = fn

    info_dict['MCC'] = (tp*tn - fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

    info_file.write(f"{label_names[do_class]},{info_dict['MCC']}\n")


info_file.close()



