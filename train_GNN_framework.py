import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from utils.data import *
from models.GNN import GCN
from utils.config import args
from utils.helper import masked_loss, masked_acc
import random
from sklearn.model_selection import train_test_splits
from helper import train_loader


##########################################################
# %% Load Data
##########################################################
# LOAD data
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
ROIs, labels, labels_index = load_fmri_data(dataDir='data', connectivity=False)
# convert to functional connectivity
connectivity_matrices = signal_to_connectivities(ROIs[:10], kind='correlation')
# adding threshold
connectivity_matrices, _ = threshold(connectivity_matrices)
# initial and node/edge embeddings
H_0 = node_embed(connectivity_matrices, dataDir='data')
H_0 = normalize_features(H_0).to(device)
# input
sparse_adj_matrices = sym_normalize_adj(connectivity_matrices).to(device)

labels = [x if (x == "CN") else "CD" for x in labels]
# label
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
    running_loss = 0
    for batch_id, data in enumerate(train_loader(mode='train', sparse_adj_matrices, label)):
        # Preparing Data
        input_data, label_data = data
        # Feedforward
        optimizer.zero_grad()
        predict = net(input_data, label_data)


        # Compute the loss
        loss = criterion(predict, label_data)

        # Calculate gradients.
        loss.backward()

        # Minimise the loss according to the gradient.
        optimizer.step()

        running_loss += loss.item()

        if batch_id % 32 == 31:
            print("Epoch: %2d, Batch: %4d, Loss: %.3f"
                  % (epoch + 1, batch_id + 1, running_loss / 32))

net.eval()
with torch.no_grad():
    for batch in train_loader(mode='test', sparse_adj_matrices, label):
        # Get a batch and potentially send it to GPU memory.
        predict = net(input_data, label_data)


##########################################################
# %% Plot result
##########################################################
# out = net((feature, support))
# out = out[0]
# acc = masked_acc(out, test_label, test_mask)
# print('test:', acc.item())