import os
os.environ["PYTHONNOUSERSITE"] = "1"

from deepspot.utils.utils_image import get_morphology_model_and_preprocess
from deepspot.utils.utils_image import compute_mini_tiles

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import yaml
import time
import matplotlib.pyplot as plt

import pickle

print('torch.cuda.is_available() is', torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device is', device)

out_folder = "example_data"
white_cutoff = 200  # recommended, but feel free to explore
downsample_factor = 10 # downsampling the image used for visualisation in squidpy
model_weights = 'DeepSpot_pretrained_model_weights/Colon_HEST1K/final_model.pkl'
model_hparam = 'DeepSpot_pretrained_model_weights/Colon_HEST1K/top_param_overall.yaml'
gene_path = 'DeepSpot_pretrained_model_weights/Colon_HEST1K/info_highly_variable_genes.csv'
sample = 'ZEN38'
image_path = f'example_data/data/image/{sample}_without_fud.jpg'

with open(model_hparam, "r") as stream:
    config = yaml.safe_load(stream)

print(config)

# the model is trained on H&E images at 20x magnification and Visium with tissue area of 55um
# NB: please adapt the parameters based on your data
# n_mini_tiles is always 9
# spot_diameter and spot_distance are based on your H&E resolution
n_mini_tiles = 9 # number of non-overlaping subspots
spot_diameter = 100#config['spot_diameter'] # spot diameter
spot_distance = 100#config['spot_distance'] # distance between spots
image_feature_model = config['image_feature_model']

print(image_feature_model)

genes = pd.read_csv(gene_path)
selected_genes_bool = genes.isPredicted.values
genes_to_predict = genes[selected_genes_bool]
genes_to_predict.sort_values("highly_variable_rank")


# Force the unpickler to find the DeepSpot class where it currently lives
import sys
from deepspot.spot import model as actual_model
from deepspot.spot import loss as actual_loss
# Force the unpickler to find 'deepspot.model' at its new home in 'deepspot.spot.model'
sys.modules['deepspot.model'] = actual_model
# Also map 'deepspot.spot' if the unpickler is looking for attributes directly there
sys.modules['deepspot.spot'] = actual_model
sys.modules['deepspot.loss'] = actual_loss
model_expression = torch.load(model_weights, map_location=device)
model_expression.to(device)
model_expression.eval()

image_feature_model_path = '/data/aggarwalm4/sensx_compare_morpho_to_spatial/UNI/pytorch_model.bin'


morphology_model, preprocess, feature_dim = get_morphology_model_and_preprocess(model_name=image_feature_model,
                                                                                device=device, model_path=image_feature_model_path)

morphology_model.to(device)
morphology_model.eval()

import numpy as np
from sklearn.neighbors import NearestNeighbors

def detach_and_convert(data):
    return data[None, ].detach().float()


resize_shape = (9, 100, 100, 3)

debug_mode = 0

import qoi_wrapper_VALIDATE as qoi_wrapper

mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

output_idxs = np.arange(5000)

if debug_mode:
    qoi_function = qoi_wrapper.qoi_wrapper_debug(resize_shape\
                               , model_expression\
                               , preprocess\
                               , compute_mini_tiles\
                               , detach_and_convert\
                               , morphology_model\
                               , mean\
                               , std\
                               , device\
                                , output_idxs\
                              )
else:
    qoi_function = qoi_wrapper.qoi_wrapper(resize_shape\
                           , model_expression\
                           , preprocess\
                           , compute_mini_tiles\
                           , detach_and_convert\
                           , morphology_model\
                           , mean\
                           , std\
                           , device\
                            , output_idxs\
                          )


if not debug_mode:
    fname = 'predicted_expression_all_spots_original.p'
else:
    fname = 'predicted_expression_all_spots_original_FORDEBUG.p'
    
dbfile = open(fname, 'rb')
info_dict = pickle.load(dbfile)
dbfile.close()


spots_processed_centers = info_dict['spots_processed_centers']
spots_processed_features = info_dict['spots_processed_features']
spots_processed_genes = info_dict['spots_processed_genes']

all_XX = []
all_gene_expr = []
all_spot_centers = []

n_spots = len(spots_processed_centers)

if debug_mode:
    spots_processed_step0 = info_dict['spots_processed_step0']
    spots_processed_step1_preprocess = info_dict['spots_processed_step1_preprocess']
    spots_processed_step2_morphologymodel = info_dict['spots_processed_step2_morphologymodel']
    spots_processed_step3_detachconvert = info_dict['spots_processed_step3_detachconvert']
    all_s0 = []
    all_s1 = []
    all_s2 = []
    all_s3 = []


BATCH_SIZE = 1000

for idx in range(n_spots):

    sc = info_dict['spots_processed_centers'][idx]
    sf = info_dict['spots_processed_features'][idx]
    sg = info_dict['spots_processed_genes'][idx]
    
    
    X_spot, X_neighbors = sf
    if len(X_neighbors) < 8:
        continue

    # Combine spot (list of 1) and neighbors (list of 8) into a single list of 9 arrays
    combined_list = [X_spot] + X_neighbors 
    
    # Stack them into shape (9, 100, 100, 3)
    stacked = np.stack(combined_list, axis=0)
    
    # Flatten to (270000,)
    this_XX = stacked.flatten()

    all_XX.append(this_XX)
    all_gene_expr.append(sg)
    all_spot_centers.append(sc)

    if debug_mode:
        s0 = info_dict['spots_processed_step0'][idx]
        s1 = info_dict['spots_processed_step1_preprocess'][idx]
        s2 = info_dict['spots_processed_step2_morphologymodel'][idx]
        s3 = info_dict['spots_processed_step3_detachconvert'][idx]
        all_s0.append(s0)
        all_s1.append(s1)
        all_s2.append(s2)
        all_s3.append(s3)

    if len(all_spot_centers) == BATCH_SIZE:
        break
    
all_XX = np.vstack(all_XX)
all_gene_expr = np.vstack(all_gene_expr)

print('all_XX has shape:', all_XX.shape)


# for ii in range(10):

tic = time.time()
if debug_mode:
    batched_preds = qoi_function(all_XX[:BATCH_SIZE], all_s0, all_s1, all_s2, all_s3, all_gene_expr)
else:
    batched_preds = qoi_function(all_XX[:BATCH_SIZE])

toc = time.time()

print(f'Time taken {toc-tic}')

print('Max discrepancy in predicted gene expression', np.amax(np.abs(batched_preds.detach().cpu().numpy() - all_gene_expr)))

# plt.scatter(batched_preds.detach().cpu().numpy().flatten(), all_gene_expr.flatten(), alpha=0.2)
# plt.xlabel('original', fontsize=16)
# plt.ylabel('batched', fontsize=16)
# plt.savefig('gene_expre_batched_vs_original.jpg', dpi=300)



# info_dict = dict()
# info_dict['all_XX'] = all_XX
# info_dict['all_spot_centers'] = all_spot_centers
# info_dict['all_gene_expr'] = batched_preds

# fname = 'predicted_genes_for_sensx.p'
# dbfile = open(fname, 'wb')
# pickle.dump(info_dict, dbfile)
# dbfile.close()


