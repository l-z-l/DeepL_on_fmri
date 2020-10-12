import os.path
import random

import bct
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import SubsetRandomSampler
from torch_geometric.data import Dataset
from torch import nn
from torch import optim

from models.GNN import GNN, GNN_SAG
from utils.data import load_fmri_data, signal_to_connectivities, node_embed, \
    row_normalize, sym_normalize, list_2_tensor, bingge_norm_adjacency
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
from torch_geometric.data import Data, DataLoader
from utils.config import args
from utils.helper import num_correct, plot_train_result, plot_evaluation_matrix
from datetime import datetime
from torch_geometric.nn import GNNExplainer

##########################################################
# %% Meta
###############train_test_split###########################
SAVE = True
MODEL_NANE = f'SAG_{datetime.now().strftime("%Y-%m-%d-%H:%M")}'
datadir = './data/'
outdir = './outputs'
dataset_name = '271_100_5_sliced_AAL'
if SAVE:
    save_path = os.path.join(outdir, f'{MODEL_NANE}_{dataset_name}/') if SAVE else ''
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
else:
    save_path = ''
##########################################################
# %% Load Data
###############train_test_split###########################
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

ROIs, labels, labels_index = load_fmri_data(dataDir=datadir, dataset=dataset_name)
labels = [x if (x == "CN") else "CD" for x in labels]

_, labels_index, classes_count = np.unique(labels, return_inverse=True, return_counts=True)
label = torch.as_tensor(labels_index, dtype=torch.float)

connectivity_matrices = signal_to_connectivities(ROIs, kind='correlation')
partial_corr = signal_to_connectivities(ROIs, kind='partial correlation')
precision = signal_to_connectivities(ROIs, kind='precision')

### get features
graphs = []
node_embeddings = []
scaler = MinMaxScaler(feature_range=(0, 1))
for i, matrix in enumerate(connectivity_matrices):
    # node is not self connected
    # np.fill_diagonal(matrix, 0)

    ### THRESHOLD: remove WHEN abs(connectivity) < mean + 1 * std
    absmx = abs(matrix)
    percentile = np.percentile(absmx, 95)  # threshold 50 % of connections
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
    x = node_embed([matrix], mask_coord='AAL').squeeze()
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
    g.x = xs[i] # + int(label[i]) * 1000

print(f"Data: {graphs[-1]}")
print(f'Is directed: {graphs[-1].is_undirected()}')
print(f'Contains isolated nodes: {graphs[-1].contains_isolated_nodes()}')
print(f'Self Connected: {graphs[-1].contains_self_loops()}')

### sampling
train_idx, valid_idx = train_test_split(np.arange(len(graphs)),
                                        test_size=0.15,
                                        shuffle=True)
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(graphs, batch_size=64, sampler=train_sampler)
test_loader = DataLoader(graphs, batch_size=64, sampler=valid_sampler)

##########################################################
# %% initialise model and loss func
##########################################################
print("--------> Using ", device)
# model = GNN(hidden_channels=64, num_node_features=x.shape[1], num_classes=2).to(device)
model = GNN_SAG(num_features=x.shape[1], nhid=64, num_classes=1).to(device)  #

optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=args.weight_decay)
criterion = nn.BCEWithLogitsLoss().to(device)

train_loss_list, test_loss_list, training_acc, testing_acc = [], [], [], []
for epoch in range(1000):
    model.train()
    train_loss, correct, total = 0, 0, 0
    val_loss, val_correct, val_total = 0, 0, 0

    ### train ###
    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        # Feedforward
        optimizer.zero_grad()
        predict = model(data.x, data.edge_index, data.edge_attr, data.batch)

        # Compute the loss
        loss = criterion(predict.squeeze(), data.y)
        loss.backward()
        optimizer.step()

        correct += num_correct(predict, data.y)
        total += len(data.y)
        train_loss += loss.item()

    train_loss_list.append(train_loss / total)
    training_acc.append(int(correct) / total * 100)

    ### test ###
    model.eval()
    with torch.no_grad():
        for test_data in test_loader:  # Iterate in batches over the training dataset.
            test_data = test_data.to(device)

            val_predict = model(test_data.x, test_data.edge_index, test_data.edge_attr, test_data.batch)
            val_correct += num_correct(val_predict, test_data.y)

            val_total += len(test_data.y)
            val_loss += criterion(val_predict.squeeze(), test_data.y).item()

    test_loss_list.append(val_loss / val_total)
    testing_acc.append(int(val_correct) / val_total * 100)

    if epoch % 50 == 0:
        print(f"====>Training: Epoch: {epoch}, Train loss: {train_loss_list[-1]:.3f}, Accuracy: {training_acc[-1]:.3f}")
        print(f"Test loss: {test_loss_list[-1]:.3f}, Accuracy: {testing_acc[-1]:.3f}")
        # print(f"Epoch: {epoch}, Loss: {running_loss/total}")

history = {
    "train_loss": train_loss_list,
    "train_acc": training_acc,
    "test_loss": test_loss_list,
    "test_acc": testing_acc,
}
history = pd.DataFrame(history)
### save
if SAVE:
    # SAVE TRAINED MODEL and history
    history.to_csv(save_path + 'epochs.csv')
    # save model
    torch.save(model, save_path + 'GCN.pth')

# %%
# node_idx = 10
# explainer = GNNExplainer(model, epochs=1000)
# node_feat_mask, edge_mask = explainer.explain_node(node_idx, data.x, data.edge_index, (data.edge_attr, data.batch))
# ax, G = explainer.visualize_subgraph(node_idx, edge_index, edge_mask, y=data.y)# %% Plot result
# plt.show()#########################################################

# %%
plot_train_result(history, save_path=save_path)

#########################################################
# %% Evaluate result
#########################################################
### test ###
# model.eval()
# label_truth = []
# label_pred = []
# with torch.no_grad():
#     for test_data in test_loader:
#         label_truth.append(test_data.y.numpy().tolist())
#         test_data = test_data.to(device)
#
#         val_predict = model(test_data.x, test_data.edge_index, test_data.edge_attr, test_data.batch)
#
#         pred = val_predict.max(dim=-1)[-1] if val_predict.shape[1] > 1 else val_predict > 0.5
#
#         label_pred.append(pred.cpu().numpy().tolist())
#
# plot_evaluation_matrix(label_truth, label_pred, save_path)
