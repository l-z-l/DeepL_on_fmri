import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import numpy as np
from utils.data import *
from models import GNN
from utils import config
from utils.helper import masked_loss, masked_acc


##########################################################
# %% Load Data
##########################################################
# LOAD data
ROIs, labels, labels_idex = load_fmri_data(connectivity=False)
# convert to functional connectivity
connectivity_matrices = signal_to_connectivities(ROIs[:10], kind='correlation')
# adding threshold
connectivity_matrices, _ = threshold(connectivity_matrices)
# inital and node/edge embeddings
H_0 = node_embed(connectivity_matrices)
H_0 = normalize_features(H_0)
W_0 = torch.as_tensor(connectivity_matrices)
sparse_adj_matrices = sym_normalize_adj(connectivity_matrices)

##########################################################
# %% initialise mode and
##########################################################
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
print("--------> Using ", device)

net = GNN(feat_dim, num_classes, num_features_nonzero)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.01)

net.train()
for epoch in range(args.epochs):

    out = net((feature, support))
    out = out[0]
    loss = masked_loss(out, train_label, train_mask)
    loss += args.weight_decay * net.l2_loss()

    acc = masked_acc(out, train_label, train_mask)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(epoch, loss.item(), acc.item())

net.eval()

##########################################################
# %% Plot result
##########################################################
out = net((feature, support))
out = out[0]
acc = masked_acc(out, test_label, test_mask)
print('test:', acc.item())