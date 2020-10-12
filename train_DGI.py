import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

from utils.data import *
from models.GNN import DGI
from utils.config import args
from utils.helper import masked_loss, masked_acc
import random
from sklearn.model_selection import train_test_splits
from utils.helper import train_loader_graph
# plot
import matplotlib.pyplot as plt

##########################################################
# %% Load Data
##########################################################
# LOAD data
dataset = '273_MSDL'
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
ROIs, labels, labels_index = load_fmri_data(dataDir='data', dataset=dataset)
# convert to functional connectivity
connectivity_matrices = signal_to_connectivities(ROIs, kind='correlation')
# adding threshold
connectivity_matrices, _ = threshold(connectivity_matrices)

### inital and node/edge embeddings
H_0 = node_embed(connectivity_matrices, 'MSDL')
H_0 = Variable(normalize_features_list(H_0), requires_grad=False).to(device)
# torch.save(H_0, "./data/273_MSDL_node.pt")
# H_0 = torch.load(f"./data/{dataset}_node.pt")
# H_0 = torch.randn((connectivity_matrices.shape[0], connectivity_matrices.shape[1], 50))

sparse_adj_list = sym_normalize_list(connectivity_matrices)

labels = [x if (x == "CN") else "CD" for x in labels]
classes, labels_index, classes_count = np.unique(labels, return_inverse=True, return_counts=True)
label = torch.as_tensor(labels_index, dtype=torch.float)
##########################################################
# %% initialise mode and
##########################################################
nb_nodes = H_0.shape[1]
batch_size = 64

net = DGI(H_0.shape[2], 200, nn.PReLU()).to(device)
optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
criterion = nn.BCEWithLogitsLoss().to(device)

net.train()
loss_values = []
for epoch in range(200):
    running_loss = 0
    correct = 0
    total = 0
    for batch_id, data in enumerate(train_loader_graph(mode='train', input=sparse_adj_list, target=label, feature=H_0)()):
        # Preparing Data
        adj, _, feat_data = data
        adj = adj.to(device)
        feat_data = feat_data.to(device)

        # Feedforward
        optimizer.zero_grad()

        idx = np.random.permutation(nb_nodes)
        shuf_fts = feat_data[:, idx, :]

        lbl_1 = torch.ones(feat_data.shape[0], nb_nodes)
        lbl_2 = torch.zeros(feat_data.shape[0], nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1).to(device)

        predict = net(feat_data, shuf_fts, adj, True, None, None, None)

        # Compute the loss
        loss = Variable(criterion(predict, lbl), requires_grad=True)
        # F.nll_loss(out, label_data)
        # Calculate gradients.
        loss.backward()
        optimizer.step()
    loss_values.append(loss.item())
    print(f'Num epochs: {epoch}, Loss: {loss.item()}')

print('Finished Training Trainset')
plt.plot(np.array(loss_values), label = "Training Loss function")
plt.xlabel('Number of epoches')
plt.title('Loss value')
plt.legend()
# plt.savefig('loss.png')
plt.show()