import os.path
import random

import bct
import torch
import matplotlib.pyplot as plt
from nilearn.connectome import ConnectivityMeasure
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import SubsetRandomSampler
from torch_geometric.data import Dataset
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GraphConv, GCNConv

from models.GNN import GNN, GNN_SAG
from utils.data import load_fmri_data, signal_to_connectivities, node_embed, \
    row_normalize, sym_normalize, list_2_tensor, bingge_norm_adjacency
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
from torch_geometric.data import Data, DataLoader
# from utils.config import args
from utils.helper import num_correct, plot_evaluation_matrix
from datetime import datetime
from torch_geometric.nn import GNNExplainer
from torch_geometric.utils import subgraph

from nilearn import plotting, datasets

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.graph_objects as go
import plotly.offline

import seaborn as sns

##########################################################
# %% Meta
###############inference_test_split###########################
SAVE = False
MODEL_NANE = f'SAG_{datetime.now().strftime("%Y-%m-%d-%H:%M")}'
datadir = './data'
outdir = './outputs'
dataset_name = '273_MSDL'
if SAVE:
    save_path = os.path.join(outdir, f'{MODEL_NANE}_{dataset_name}/') if SAVE else ''
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
else:
    save_path = ''

##########################################################
# %% Load Data
###############inference_test_split###########################
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

ROIs, labels, labels_index = load_fmri_data(dataDir=datadir, dataset=dataset_name)

# labels = np.concatenate(labels)
# take the frist 20
labels = [x if (x == "CN") else "CD" for x in labels]

_, labels_index, classes_count = np.unique(labels, return_inverse=True, return_counts=True)
label = torch.as_tensor(labels_index, dtype=torch.float)

connectivity_matrices = signal_to_connectivities(ROIs, kind='correlation')
partial_corr = signal_to_connectivities(ROIs, kind='partial correlation')
precision = signal_to_connectivities(ROIs, kind='precision')

# %%
### get features
graphs = []
node_embeddings = []
scaler = MinMaxScaler(feature_range=(0, 1))
for i, matrix in enumerate(connectivity_matrices):
    # node is not self connected
    # np.fill_diagonal(matrix, 0)

    ### THRESHOLD: remove WHEN abs(connectivity) < mean + 1 * std
    absmx = abs(matrix)
    percentile = np.percentile(absmx, 5)  # threshold 50 % of connections
    # mean, std = np.mean(abs(matrix)), np.std(abs(matrix))
    mask = absmx < percentile
    # mask = (mean + 0.5 * std)

    # apply mask to edge_attr
    matrix[mask] = 0
    partial_corr[i][mask] = 0
    precision[i][mask] = 0

    ### convert to binary matrix
    # made a distinct adj matrix, since connection weights are counted in edge attr
    matrix[matrix != 0] = 1

    ### edge_attr
    corr = sp.coo_matrix(matrix)
    par_corr = sp.coo_matrix(partial_corr[i])
    covar = sp.coo_matrix(precision[i])
    edge_attr = torch.from_numpy(np.vstack((corr.data, par_corr.data, covar.data)).transpose())

    ### node_embed
    x = node_embed([matrix], mask_coord='MSDL').squeeze()
    # print(x[0]) # TODO: check later
    # TODO: Node normalizaiton
    node_embeddings.append(x)
    # x = torch.from_numpy(normalize(x))

    ### normalise graph adj
    edge_index = bingge_norm_adjacency(matrix)
    edge_index = torch.from_numpy(np.vstack((edge_index.row, edge_index.col))).long()

    graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label[i:i + 1]))

xs = torch.cat(node_embeddings, 0)
xs = scaler.fit_transform(xs)
xs = torch.tensor(xs.reshape((ROIs.shape[0], ROIs.shape[2], xs.shape[-1])), dtype=torch.float)

# normalize edge
for i, g in enumerate(graphs):
    g.x = xs[i]  # + int(label[i]) * 10000

print(f"Data: {graphs[-1]}")
print(f'Is directed: {graphs[-1].is_undirected()}')
print(f'Contains isolated nodes: {graphs[-1].contains_isolated_nodes()}')
print(f'Self Connected: {graphs[-1].contains_self_loops()}')

### sampling
inference_idx, valid_idx = train_test_split(np.arange(len(graphs)),
                                            test_size=0.1,
                                            shuffle=True, random_state=None)
inference_sampler = SubsetRandomSampler(inference_idx)

inference_loader = DataLoader(graphs, batch_size=128, sampler=inference_sampler)
##########################################################
# %% train
##########################################################
print("--------> Using ", device)
# load the model
model = GNN_SAG(num_features=x.shape[1], nhid=10, num_classes=2, pooling_ratio=0.5,
                dropout_ratio=0.5).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)
criterion = nn.CrossEntropyLoss().to(device)

train_loss_list, test_loss_list, training_acc, testing_acc = [], [], [], []
for epoch in range(500):
    model.train()
    train_loss, correct, total = 0, 0, 0
    val_loss, val_correct, val_total = 0, 0, 0

    ### train ###
    for data in inference_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        # Feedforward
        optimizer.zero_grad()
        predict = model(data.x, data.edge_index, data.edge_attr, data.batch)

        # Compute the loss
        loss = criterion(predict.squeeze(), data.y.long())
        loss.backward()
        optimizer.step()

        correct += num_correct(predict, data.y)
        total += len(data.y)
        train_loss += loss.item()
    train_loss_list.append(train_loss / total)
    training_acc.append(int(correct) / total * 100)

    if epoch % 50 == 0:
        print(f"====>Training: Epoch: {epoch}, Train loss: {train_loss_list[-1]:.3f}, Accuracy: {training_acc[-1]:.3f}")
        # print(f"Epoch: {epoch}, Loss: {running_loss/total}")


##########################################################
# %% initialise model and loss func
##########################################################
# model.load_state_dict(torch.load('./outputs/GAT_273_MSDL/GCN.pth'))

### test ###
model.eval()
embedded = []
label = []
with torch.no_grad():
    for data in inference_loader:  # Iterate in batches over the training dataset.
        label.append(data.y)
        data = data.to(device)
        val_predict = model(data.x, data.edge_index, data.edge_attr, data.batch)
        embedded.append(val_predict.detach().cpu())
        # embedded.append(val_predict.detach().cpu())

embedded_points = np.concatenate(embedded, axis=0)

# tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
# tsne_pca_results = tsne.fit_transform(embedded_points)
### PCA
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(embedded_points)

df = pd.DataFrame(tsne_pca_results, columns=['tsne_1', 'tsne_2'])
df['y'] = pd.Series(np.concatenate(label, axis=0))

sns.scatterplot(
    x="tsne_1", y="tsne_2",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=df,
    legend="full",
    alpha=0.5
)
plt.show()
plt.savefig()

'''
#########################################################
# %% Interpret result
#########################################################
# load the model
model = GNN_SAG(num_features=x.shape[1], nhid=10, num_classes=2, pooling_ratio=0.5,
            dropout_ratio=0.5)# .to(device)
model.load_state_dict(torch.load('./outputs/GAT_273_MSDL/GCN.pth'))

val_loader = DataLoader(graphs, batch_size=1, sampler=valid_sampler)
valiter = iter(val_loader)
data_batch = next(valiter)# .to(device)

# %%
# only keep the one edge attr as edge weights
data = data_batch.to_data_list()[0]
data.edge_attr = data.edge_attr[:, 1]
raw_networkx = to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'], to_undirected=True, remove_self_loops=True)
raw_adj = nx.to_numpy_array(raw_networkx, weight='edge_attr') # plot to connectome

# %%
adj, weight = data_batch.edge_index, data_batch.edge_attr
param_dict = model.interpret(data_batch.x, adj, weight, data_batch.batch)
'''
