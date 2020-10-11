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
# H_0 = node_embed(connectivity_matrices, 'Havard_Oxford')
# H_0 = Variable(normalize_features(H_0), requires_grad=False).to(device)
# torch.save(H_0, "./data/273_MSDL_node.pt")
H_0 = torch.load(f"./data/{dataset}_node.pt")
H_0 = torch.zeros((connectivity_matrices.shape[0], connectivity_matrices.shape[1], 20))

sparse_adj_list = sym_normalize_adj(connectivity_matrices)

labels = [x if (x == "CN") else "CD" for x in labels]
classes, labels_index, classes_count = np.unique(labels, return_inverse=True, return_counts=True)
label = torch.as_tensor(labels_index, dtype=torch.float)

##########################################################
# %% initialise mode and
##########################################################
print("--------> Using ", device)
model = GCN(H_0.shape[2], 2, node_num=H_0.shape[1])
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
criterion = torch.nn.NLLLoss().to(device)

train_loss_list, val_loss_list, training_acc, testing_acc = [], [], [], []
for epoch in range(1000):
    train_loss, correct, total = 0, 0, 0
    val_loss, val_correct, val_total = 0, 0, 0

    ### train ###
    for batch_id, data in enumerate(train_loader_graph(mode='train', input=sparse_adj_list, target=label, feature=H_0)()):
        model.train()
        # Preparing Data
        input_data, label_data, feat_data = data
        input_data, label_data, feat_data = input_data.to(device), label_data.to(device), feat_data.to(device)
        # Feedforward
        optimizer.zero_grad()

        predict = model((feat_data, input_data))
        # Compute the loss
        loss = criterion(predict, label_data.long())
        loss.backward()
        optimizer.step()

        ### calculate loss
        pred = predict.max(dim=-1)[-1]
        correct += pred.eq(label_data).sum().item()
        total += len(label_data)

        train_loss += loss.item()

    ### test ###
    model.eval()
    with torch.no_grad():
        for val_batch_id, val_data in enumerate(train_loader_graph(mode='train', input=sparse_adj_list, target=label, feature=H_0)()):
            val_x, val_y, val_feat = val_data
            val_x, val_y, val_feat = val_x.to(device), val_y.to(device), val_feat.to(device)

            val_predict = model((val_feat, val_x))
            val_pred = val_predict.max(dim=-1)[-1]
            val_correct += val_pred.eq(val_y).sum().item()

            val_total += len(val_y)
            val_loss += criterion(predict, label_data.long()).item()

    val_loss_list.append(val_loss)
    train_loss_list.append(train_loss)
    training_acc.append(int(correct)/total * 100)
    testing_acc.append(int(val_correct)/val_total * 100)

    if epoch % 10 == 0:
        print(f"====>Training: Epoch: {epoch}, Train loss: {train_loss}, Accuracy: {training_acc[-1]}")
        print(f"Test loss: {val_loss}, Accuracy: {testing_acc[-1]}")
        # print(f"Epoch: {epoch}, Loss: {running_loss/total}")

#########################################################
# %% Plot result
#########################################################
print('Finished Training Trainset')
plt.plot(np.array(train_loss_list), label="Training Loss function")
plt.plot(np.array(val_loss_list), label="Testing Loss function")
plt.xlabel('Number of epoches')
plt.title('Loss value')
plt.legend()
# plt.savefig('loss.png')
plt.show()

print('Finished Testing Trainset')
plt.plot(np.array(training_acc), label="Train Accuracy")
plt.plot(np.array(testing_acc), label="Test Accuracy")
plt.xlabel('Number of epoches')
plt.title('Accuracy')
plt.legend()
# plt.savefig('accuracy.png')
plt.show()