import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

from utils.data import *
from models.CNN import SpatialTemporalCNN
from utils.config import args
from utils.helper import train_vec_loader, train_loader

from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
# plot
import matplotlib.pyplot as plt

##########################################################
# %% Load Data
##########################################################
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
ROIs, labels, labels_index = load_fmri_data(dataDir='data', dataset='interpolation/270_every_10_MAX_sliced_AAL')
# convert to functional connectivity
X = torch.as_tensor(ROIs, dtype=torch.float)
X = torch.unsqueeze(X, 1).to(device) # add extra dimension (m, 1, time_seq, ROIS)

labels = [x if (x == "CN") else "CD" for x in labels]
classes, labels_index, classes_count = np.unique(labels, return_inverse=True, return_counts=True)
label = torch.as_tensor(labels_index, dtype=torch.float)
##########################################################
# %% initialise mode and
##########################################################
model = SpatialTemporalCNN()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

loss_values, testing_acc = [], []

model.train()
criterion = torch.nn.CrossEntropyLoss().to(device)

loss_values, testing_acc = [], []

model.train()

for epoch in range(200):
    running_loss = 0
    correct = 0
    total = 0
    for batch_id, data in enumerate(train_vec_loader(batch_size=50, mode='train', input=X, target=label)()):
        # Preparing Data
        input_data, label_data = data
        input_data = input_data.to(device)
        # Feedforward
        optimizer.zero_grad()

        predict = model(input_data)

        out = torch.squeeze(predict.detach().cpu())
        # pred = out > 0.5
        # correct += (pred == label_data).sum()
        pred = out.max(dim=-1)[-1]
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

'''
# testing
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in train_vec_loader(mode='test', input=X, target=label)():
        input_data, label_data, feat_data = batch
        input_data = input_data.to(device)
        feat_data = feat_data.to(device)
        # Get a batch and potentially send it to GPU memory.
        predict = model((feat_data, input_data))

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
plt.plot(np.array(loss_values), label="Training Loss function")
plt.xlabel('Number of epoches')
plt.title('Loss value')
plt.legend()
# plt.savefig('loss.png')
plt.show()
'''
print('Finished Testing Trainset')
plt.plot(np.array(testing_acc), label="Accuracy function")
plt.xlabel('Number of epoches')
plt.title('Accuracy')
plt.legend()
plt.savefig('accuracy.png')
plt.show()
'''