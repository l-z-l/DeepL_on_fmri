import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

from utils.data import *
from models.MLP import Linear
from utils.config import args
from utils.helper import train_vec_loader

from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
# plot
import matplotlib.pyplot as plt

##########################################################
# %% Load Data
##########################################################
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
ROIs, labels, labels_index = load_fmri_data(dataDir='data', dataset='273_Havard_Oxford')
# convert to functional connectivity
connectivity_matrices = signal_to_connectivities(ROIs, kind='correlation', discard_diagonal=True, vectorize=True)
X = torch.as_tensor(connectivity_matrices, dtype=torch.float)

labels = [x if (x == "CN") else "CD" for x in labels]
classes, labels_index, classes_count = np.unique(labels, return_inverse=True, return_counts=True)
label = torch.as_tensor(labels_index, dtype=torch.float)
##########################################################
# %% initialise mode and
##########################################################

print("--------> Using ", device)
model = Linear(X.shape[1], 1)
model.to(device)
# optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=args.weight_decay)

# loss = torch.nn.BCELoss().to(device)

criterion = torch.nn.BCELoss().to(device)

loss_values, testing_acc = [], []

##########################################################
# %% Artificial Data
##########################################################
# positive_index = []
# for index, num in enumerate(label):
#     if num == 1:
#         positive_index.append(index)
# # Taking positive sample from X
# # X_1 = X
# print(positive_index)
# for index, num in enumerate(X):
#     if index in positive_index:
#         # print(f"{index} {X[index]} {X[index].shape}")
#         X[index] = X[index] + 10
        # print(f"{index} {X[index]} {X[index].shape}")

# no need to do batch, since dataset size is small (79)
loss_values = []
acc_values = []


testing_loss = []
testing_acc = []

# input_1 = []
for epoch in range(100):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    running_loss_test = 0
    correct_test = 0
    total_test = 0
    for batch_id, data in enumerate(train_vec_loader(batch_size=50, mode='train', input=X, target=label)()):
        # Preparing Data
        input_data, label_data = data
        input_data = input_data.to(device)
        # Feedforward
        # print()
        optimizer.zero_grad()

        predict = model(input_data)

        # out = torch.squeeze(predict) #.detach().cpu())
        ### BCE
        pred = predict > 0.5
        correct += (torch.squeeze(pred) == label_data).sum()

        ### Cross Entropy
        # pred = out.max(dim=-1)[-1]
        # out = out.squeeze()
        # correct += pred.eq(label_data).sum().item()

        total += len(label_data)

        # Compute the loss
        loss = criterion(predict, label_data) #, requires_grad=True)
        # loss = criterion(out, label_data.float())
        # F.nll_loss(out, label_data)
        # Calculate gradients.
        loss.backward()
        # Minimise the loss according to the gradient.
        optimizer.step()

        # running_loss = (loss / len(input_data))
        running_loss += loss.item()

    loss_values.append(loss.item())
    testing_acc.append(int(correct)/total * 100)
    print(f"Training ---- Epoch: {epoch: <10} Loss: {running_loss/total: ^10} correct: {correct: ^10} total: {total: ^10}"
          f"Accuracy: {int(correct)/total * 100: >5}")
    # print(f"Epoch: {epoch}, Loss: {running_loss}, Loss_1: {running_loss_1}")
    model.eval()
    for batch_id, data in enumerate(train_vec_loader(batch_size=50, mode='test', input=X, target=label)()):
        input_data, label_data = data
        input_data = input_data.to(device)
        with torch.no_grad():
            predict = model(input_data)
            ### BCE
            pred = predict > 0.5
            loss = criterion(predict, label_data)
            running_loss_test += loss.item()
            correct_test += (torch.squeeze(pred) == label_data).sum()
            total_test += len(label_data)
    testing_loss.append(running_loss_test/total_test)
    acc_values.append(int(correct_test) / total_test * 100)
    print(f"Testing ---- Epoch: {epoch: <10} Loss: {running_loss_test/total_test: ^10} correct: {correct_test: ^10} total: {total_test: ^10} Accuracy: {int(correct_test)/total_test * 100: >5}")

#########################################################
# %% Plot result
#########################################################
print('Finished Training Trainset')
plt.plot(np.array(loss_values), label="Training Loss function")
plt.plot(np.array(testing_loss), label="Testing Loss function")
plt.xlabel('Number of epoches')
plt.title('Loss value')
plt.legend()
plt.savefig('loss.png')
plt.show()

print('Finished Testing Trainset')
plt.plot(np.array(acc_values), label="Training Accuracy function")
plt.plot(np.array(testing_acc), label="Testing Accuracy function")
plt.xlabel('Number of epoches')
plt.title('Accuracy')
plt.legend()
plt.savefig('accuracy.png')
plt.show()
