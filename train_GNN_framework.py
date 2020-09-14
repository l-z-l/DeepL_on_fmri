import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
from utils.data import *
from models.GNN import GCN
from utils.config import args
from utils.helper import masked_loss, masked_acc
import random
from sklearn.model_selection import train_test_splits
from utils.helper import train_loader


##########################################################
# %% Load Data
##########################################################
# LOAD data
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
ROIs, labels, labels_index = load_fmri_data(dataDir='data', connectivity=False)
# convert to functional connectivity
connectivity_matrices = signal_to_connectivities(ROIs, kind='correlation')
# adding threshold
connectivity_matrices, _ = threshold(connectivity_matrices[:100])
# inital and node/edge embeddings
H_0 = node_embed(connectivity_matrices, dataDir='data')
# H_0 = Variable(normalize_features(H_0), requires_grad=False).to(device)

sparse_adj_list = sym_normalize_adj(connectivity_matrices)
# sparse_adj_list = Variable(sparse_adj_list, requires_grad=False).to(device)

labels = [x if (x == "CN") else "CD" for x in labels]
classes, label, classes_count = np.unique(labels, return_inverse=True, return_counts=True)
label = torch.as_tensor(label[:100], dtype=torch.float)
##########################################################
# %% initialise mode and
##########################################################
print("--------> Using ", device)
net = GCN(H_0.shape[2], len(classes))
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
criterion = torch.nn.BCEWithLogitsLoss().to(device)

net.train()
for epoch in range(50):
    running_loss = 0
    correct = 0
    total = 0
    for batch_id, data in enumerate(train_loader(mode='train', input=sparse_adj_list, feature=H_0, target=label)()):
        # Preparing Data
        input_data, label_data, feat_data = data

        input_data = input_data.to(device)
        feat_data = feat_data.to(device)
        # Feedforward
        optimizer.zero_grad()

        predict = net((feat_data, input_data))

        out = torch.squeeze(predict.detach().cpu())
        pred = out > 0.5

        correct += (pred == label_data).sum()
        # print(pred)
        total += len(label_data)

        # Compute the loss
        loss = Variable(criterion(out, label_data) + args.weight_decay * net.l2_loss(), requires_grad = True)
        # Calculate gradients.
        loss.backward()
        # Minimise the loss according to the gradient.
        optimizer.step()

        running_loss += loss.item()

        # print(batch_id)
        # if batch_id % 32 == 31:
    print("Epoch: %2d, Loss: %.3f Accuracy: %.3f"
          % (epoch, running_loss / total, correct/total))

# net.eval()
# with torch.no_grad():
#     for batch in train_loader(mode='test', input=sparse_adj_list, target=label):
#         # Get a batch and potentially send it to GPU memory.
#         predict = net(input_data, label_data)

#########################################################
# %% Plot result
#########################################################
# out = net((feature, support))
# out = out[0]
# acc = masked_acc(out, test_label, test_mask)
# print('test:', acc.item())
