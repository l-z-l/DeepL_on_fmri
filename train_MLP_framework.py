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
ROIs, labels, labels_index = load_fmri_data(dataDir='data/', dataset='271_AAL')
# convert to functional connectivity
connectivity_matrices = signal_to_connectivities(ROIs, kind='tangent', discard_diagonal=True, vectorize=True)
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
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
loss = torch.nn.BCELoss().to(device)

loss_values, testing_acc = [], []

def test(X_test, y_test):
    # Test
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in zip(X_test, y_test):
            pred_probs = model(data)
            pred_probs = torch.squeeze(pred_probs.detach().cpu())
            # sum up batch loss
            test_loss = loss(pred_probs, torch.tensor([target]))
            # print(f"O:{output.view(1)} T:{target}")
            pred = pred_probs > 0.5
            correct += (pred == target).sum()

    test_loss /= len(X_test)
    acc = 100. * correct / len(X_test)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(X_test),
    #     acc))
    return acc/100, test_loss

# # Spliting the data
np.random.seed(42)

# no need to do batch, since dataset size is small (79)
loss_values = []
acc_values = []
testing_loss = []
testing_acc = []

test_size = int(X.shape[0] * 0.15)
for epoch in range(15000):
    model.train()
    optimizer.zero_grad()

    running_loss = 0.0
    total_acc = 0

    # random permutate data
    idx_batch = np.random.permutation(int(X.shape[0])) # np.array(range(0, X.shape[0])) #
    idx_batch_test = idx_batch[:int(test_size)]
    idx_batch_train = idx_batch[-int(len(X) - test_size):]
    # batch
    train_label_batch = label[idx_batch_train].to(device)
    train_data_batch = X[idx_batch_train].to(device)
    test_label_batch = label[idx_batch_test].to(device)
    test_data_batch = X[idx_batch_test].to(device)

    # train
    pred_probs = model(train_data_batch)
    y_pred = pred_probs > 0.5
    loss_val = loss(pred_probs, train_label_batch)
    loss_val.backward()
    optimizer.step()

    loss_values.append(loss_val / len(train_data_batch))

    acc = accuracy_score(train_label_batch.cpu(), y_pred.cpu())
    if epoch % 100 == 0:
        print("Training --------> Epoch: {}, Loss: {}, Accuracy: {}".format(epoch, loss_val, acc))
        print("-" * 80)
        print("Testing Epoch: {}".format(epoch))
        model.eval()
        with torch.no_grad():
            pred_prob_test = model(test_data_batch)
            y_pred_test = pred_prob_test > 0.5
            loss_val_test = loss(pred_prob_test, test_label_batch)
            acc_test = accuracy_score(test_label_batch.cpu(), y_pred_test.cpu())
            print("Epoch: {}, Loss: {}, Accuracy: {}".format(epoch, loss_val_test, acc_test))
        print("-" * 80)

'''
# testing
net.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in train_loader(mode='test', input=sparse_adj_list, feature=H_0, target=label)():
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

'''
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

