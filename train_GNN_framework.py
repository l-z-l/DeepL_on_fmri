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
# H_0 = torch.load(f"./data/{dataset}_node.pt")
H_0 = torch.zeros((connectivity_matrices.shape[0], connectivity_matrices.shape[1], 20))

sparse_adj_list = sym_normalize_adj(connectivity_matrices)

labels = [x if (x == "CN") else "CD" for x in labels]
classes, labels_index, classes_count = np.unique(labels, return_inverse=True, return_counts=True)
label = torch.as_tensor(labels_index, dtype=torch.float)

##########################################################
# %% initialise mode and
##########################################################
print("--------> Using ", device)
net = GCN(H_0.shape[2], 2, node_num=H_0.shape[1])
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
criterion = torch.nn.NLLLoss().to(device)

loss_values, testing_acc = [], []

net.train()
for epoch in range(50):
    running_loss = 0
    correct = 0
    total = 0
    for batch_id, data in enumerate(train_loader(mode='train', input=sparse_adj_list, target=label, feature=H_0)()):
        # Preparing Data
        input_data, label_data, feat_data = data
        input_data = input_data.to(device)
        feat_data = feat_data.to(device)
        # Feedforward
        optimizer.zero_grad()

        predict = net((feat_data, input_data))

        out = torch.squeeze(predict.detach().cpu())
        # pred = out > 0.5
        # correct += (pred == label_data).sum()
        pred = out.max(dim=1)[1]
        correct += pred.eq(label_data).sum().item()

        total += len(label_data)

        # Compute the loss
        loss = Variable(criterion(out, label_data.long()), requires_grad=True)
        # F.nll_loss(out, label_data)
        # Calculate gradients.
        loss.backward()
        # Minimise the loss according to the gradient.
        optimizer.step()

        running_loss += loss.item()

        # print(correct, total)
        # if batch_id % 32 == 31:
    # print("Epoch: %2d, Loss: %.3f Accuracy: %.3f"
    #       % (epoch, running_loss / total, correct, total))
    loss_values.append(loss.item())
    testing_acc.append(int(correct)/total * 100)
    print(f"Epoch: {epoch}, Loss: {running_loss/total} correct: {correct}, toal: {total}, Accuracy: {int(correct)/total * 100}")

# testing
net.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in train_loader(mode='test', input=sparse_adj_list, target=label, feature=H_0)():
        input_data, label_data, feat_data = batch
        input_data = input_data.to(device)
        feat_data = feat_data.to(device)
        # Get a batch and potentially send it to GPU memory.
        predict = net((feat_data, input_data))

        out = torch.squeeze(predict.detach().cpu())
        # pred = out > 0.5
        # correct += (pred == label_data).sum()

        pred = out.max(dim=1)[1]
        correct += pred.eq(label_data).sum().item()
        total += len(label_data)
    print(f"Correct: {correct}, total: {total}, Accuracy: {int(correct)/total * 100}")

#########################################################
# %% Plot result
#########################################################
print('Finished Training Trainset')
plt.plot(np.array(loss_values), label = "Training Loss function")
plt.xlabel('Number of epoches')
plt.title('Loss value')
plt.legend()
plt.savefig('loss.png')
plt.show()

print('Finished Testing Trainset')
plt.plot(np.array(testing_acc), label="Accuracy function")
plt.xlabel('Number of epoches')
plt.title('Accuracy')
plt.legend()
plt.savefig('accuracy.png')
plt.show()
''''''