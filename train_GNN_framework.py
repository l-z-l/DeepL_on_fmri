import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import numpy as np
from utils.data import *
from models.GNN import GCN
from utils.config import args
from utils.helper import masked_loss, masked_acc


##########################################################
# %% Load Data
##########################################################
# LOAD data
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
ROIs, labels, labels_idex = load_fmri_data(dataDir='data', connectivity=False)
# convert to functional connectivity
connectivity_matrices = signal_to_connectivities(ROIs[:10], kind='correlation')
# adding threshold
connectivity_matrices, _ = threshold(connectivity_matrices)
# inital and node/edge embeddings
H_0 = node_embed(connectivity_matrices, dataDir='data')
H_0 = normalize_features(H_0).to(device)
sparse_adj_matrices = sym_normalize_adj(connectivity_matrices).to(device)

labels = [x if (x == "CN") else "CD" for x in labels]
classes, label, classes_count = np.unique(labels, return_inverse=True, return_counts=True)
#############             #############################################
# %% initialise mode and
##########################################################
print("--------> Using ", device)
net = GCN(H_0.shape[2], len(classes))
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
criterion = torch.nn.BCEWithLogitsLoss().to(device) 

net.train()
for epoch in range(args.epochs):
    out = net((H_0, sparse_adj_matrices))
    zzz = out.shape
    print(out.shape)

    out = out.detach().cpu()
    loss = criterion(out, label)
    print()
    # loss += args.weight_decay * net.l2_loss()
    #
    # acc = masked_acc(out, train_label, train_mask)

    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    #
    # if epoch % 10 == 0:
    #     print(epoch, loss.item(), acc.item())
net.eval()

##########################################################
# %% Plot result
##########################################################
# out = net((feature, support))
# out = out[0]
# acc = masked_acc(out, test_label, test_mask)
# print('test:', acc.item())