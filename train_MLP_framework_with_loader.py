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
ROIs, labels, labels_index = load_fmri_data(dataDir='data', dataset='271_AAL')
# convert to functional connectivity
connectivity_matrices = signal_to_connectivities(ROIs, kind='correlation', discard_diagonal=True, vectorize=True)
X = torch.as_tensor(connectivity_matrices, dtype=torch.float)

labels = [x if (x == "CN") else "CD" for x in labels]
classes, labels_index, classes_count = np.unique(labels, return_inverse=True, return_counts=True)
label = torch.as_tensor(labels_index, dtype=torch.float)
##########################################################
# %% initialise model and loss func
##########################################################

print("--------> Using ", device)
model = Linear(X.shape[1], 1)
model.to(device)
# optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=args.weight_decay)

# criterion = torch.nn.BCELoss().to(device)
criterion = torch.nn.BCELoss().to(device)

train_loss_list, val_loss_list, training_acc, testing_acc = [], [], [], []
for epoch in range(500):
    model.train()
    train_loss, correct, total = 0, 0, 0
    val_loss, val_correct, val_total = 0, 0, 0

    for batch_id, data in enumerate(train_vec_loader(batch_size=128, mode='train', input=X, target=label)()):
        # Preparing Data
        input_data, label_data = data
        input_data, label_data = input_data.to(device), label_data.to(device)

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
        train_loss += loss.item()
    train_loss_list.append(train_loss/total)
    training_acc.append(int(correct)/total * 100)

    ### test
    model.eval()
    with torch.no_grad():
        for batch_id, data in enumerate(train_vec_loader(batch_size=50, mode='test', input=X, target=label)()):
            val_x, val_y = data
            val_x, val_y = val_x.to(device), val_y.to(device)

            val_predict = model(val_x)
            ### BCE Loss
            pred = val_predict > 0.5
            loss = criterion(val_predict, val_y)

            val_loss += loss.item()
            val_correct += (torch.squeeze(pred) == val_y).sum()
            val_total += len(val_y)

    val_loss_list.append(val_loss/val_total)
    testing_acc.append(int(val_correct) / val_total * 100)
    if epoch % 50 == 0:
        print(f"====>Training: Epoch: {epoch}, Train loss: {train_loss_list[-1]:.3f}, Accuracy: {training_acc[-1]:.3f}")
        print(f"Test loss: {val_loss_list[-1]:.3f}, Accuracy: {testing_acc[-1]:.3f}")

#########################################################
# %% Plot result
#########################################################
print('Finished Training Trainset')
plt.plot(np.array(train_loss_list), label="Training Loss function")
plt.plot(np.array(val_loss_list), label="Testing Loss function")
plt.xlabel('Number of epoches')
plt.title('Loss value')
plt.legend()
plt.savefig('loss.png')
plt.show()

print('Finished Testing Trainset')
plt.plot(np.array(training_acc), label="Train Accuracy")
plt.plot(np.array(testing_acc), label="Test Accuracy")
plt.xlabel('Number of epoches')
plt.title('Accuracy')
plt.legend()
plt.savefig('accuracy.png')
plt.show()
