import os.path
import random

import bct
import torch
import matplotlib.pyplot as plt
from torch import nn

import nilearn
from nilearn import plotting, datasets
from nilearn.regions import connected_regions
from nilearn.image import mean_img, index_img, threshold_img, get_data, load_img


from models.GNN import GNN, GNN_SAG
from utils.data import load_fmri_data, signal_to_connectivities, node_embed, \
    row_normalize, sym_normalize, list_2_tensor, bingge_norm_adjacency
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
from torch_geometric.data import Data, DataLoader
# from utils.config import args
from utils.helper import num_correct, plot_train_result, plot_evaluation_matrix
from datetime import datetime
from torch_geometric.nn import GNNExplainer

##########################################################
# %% Meta
###############train_test_split###########################

MODEL_NANE = f'SAG_{datetime.now().strftime("%Y-%m-%d-%H:%M")}'
datadir = './data'
outdir = './outputs'
dataset_name = '273_MSDL'

##########################################################
# %% Load Data
###############train_test_split###########################

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

ROIs, labels, labels_index = load_fmri_data(dataDir=datadir, dataset=dataset_name)

# functional connectivity
functional_connectivity = signal_to_connectivities(ROIs)
labels = [x if (x == "CN") else "CD" for x in labels]

matrix = functional_connectivity[0]

# %%
### coordinates
coordinates = np.load(f'./data/MSDL_coordinates.npy', allow_pickle=True)
# or find coordiantes
aal_atlas = datasets.fetch_atlas_aal(version='SPM12', data_dir=None, url=None, resume=True, verbose=1)
maps = nilearn.image.load_img(aal_atlas['maps'], wildcards=True, dtype=None)
'''
# %%
coordinates = nilearn.plotting.find_parcellation_cut_coords(maps)

# %%
### view ROI of Atlas
roi_plot = plotting.plot_roi(aal_atlas['maps'], title="AAL")
cut_coords = roi_plot.cut_coords

### view epi
plotting.plot_epi(aal_atlas['maps'], cut_coords=cut_coords,
                  title='Epi image',
                  vmax=np.max(get_data(aal_atlas['maps'])),
                  vmin=np.min(get_data(aal_atlas['maps']))
                  )
plt.show()

# %%
### view connectome 2D
plotting.plot_connectome(matrix, coordinates, edge_threshold="99.5%", node_size=20, colorbar=True)
plt.show()

# %%
### view connectome strength
# plot the positive part of of the matrix
plotting.plot_connectome_strength(
    np.clip(matrix, 0, matrix.max()), coordinates, node_size='auto', cmap=plt.cm.YlOrRd,
    title='Strength of the positive edges of the Power correlation matrix'
)

# plot the negative part of of the matrix
plotting.plot_connectome_strength(
    np.clip(matrix, matrix.min(), 0), coordinates, node_size='auto', cmap=plt.cm.PuBu,
    title='Strength of the negative edges of the Power correlation matrix'
)
plt.show()
'''
'''
# %%
### connectome 3D
view = plotting.view_connectome(matrix, coordinates, edge_threshold='95%')
view.open_in_browser()
'''
# %%
### add markers
# https://nilearn.github.io/auto_examples/03_connectivity/plot_seed_to_voxel_correlation.html#sphx-glr-auto-examples-03-connectivity-plot-seed-to-voxel-correlation-py
# display.add_markers

# %%
AD = load_img('./data/sample_raw_data/AD_2012-11-08.nii')
CN = load_img('./data/sample_raw_data/CN_2012-03-01.nii')
threshold_percentile_AD = threshold_img(AD, threshold='95%', copy=False)
threshold_percentile_CN = threshold_img(AD, threshold='95%', copy=False)

regions_percentile_img_AD, index = mean_img(connected_regions(threshold_percentile_AD,
                                                  min_region_size=1500))
for mode in ['x']: #, 'y', 'z']:
    plot_regions_AD = plotting.plot_prob_atlas(regions_percentile_img_AD, bg_img=mean_img(AD),
                             view_type='contours', display_mode=mode,
                             cut_coords=5, title="title")
    plot_regions_AD.savefig(f'./outputs/regions_AD_{mode}.png')

# plot_threshold_percentile_AD = plotting.plot_stat_map(mean_img(threshold_percentile_AD), display_mode='y', cut_coords=10,
#                        title='Threshold image with string percentile', colorbar=False)
# plot_threshold_percentile_AD.savefig('./outputs/threshold_percentile_AD_y.png')
#
# plot_threshold_percentile_CN = plotting.plot_stat_map(mean_img(threshold_percentile_CN), display_mode='y', cut_coords=10,
#                        title='Threshold image with string percentile', colorbar=False)
# plot_threshold_percentile_CN.savefig('./outputs/threshold_percentile_CN_y.png')

