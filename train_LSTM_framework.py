import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

from utils.data import *
from models.LSTM import LSTM
from utils.config import args
from utils.helper import train_vec_loader, train_loader, plot_train_result

from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
# plot
import matplotlib.pyplot as plt

##########################################################
# %% Load Data
##########################################################
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
ROIs, labels, labels_index = load_fmri_data(dataDir='data', dataset='271_100_5_sliced_AAL')
# interpolation/Havard_Oxford
# convert to functional connectivity
# Only use with AAL
# ROIs[:, :, :90]
X = torch.as_tensor(ROIs, dtype=torch.float)
# X = torch.unsqueeze(X, 1).to(device) # add extra dimension (m, 1, ROI, time_seq)
# X = x.permute # 271 * 140 * 116
# X = X[:3]
labels = [x if (x == "CN") else "CD" for x in labels]
classes, labels_index, classes_count = np.unique(labels, return_inverse=True, return_counts=True)
label = torch.as_tensor(labels_index, dtype=torch.float)
##########################################################
# %% initialise mode and
##########################################################
model = LSTM(input_size=X.shape[2], hidden_dim=16, seq_len=X.shape[1], num_layers=1, output_size=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss().to(device)

train_loss_list, val_loss_list, training_acc, testing_acc = [], [], [], []
for epoch in range(100):
    model.train()
    train_loss, correct, total = 0, 0, 0
    val_loss, val_correct, val_total = 0, 0, 0

    ### train ###
    for batch_id, data in enumerate(train_vec_loader(batch_size=64, mode='train', input=X, target=label)()):
        # Preparing Data
        input_data, label_data = data
        input_data, label_data = input_data.to(device), label_data.to(device)
        # Feedforward
        optimizer.zero_grad()
        predict = model(input_data)
        # Compute the loss
        loss = criterion(predict, label_data.long())
        loss.backward()
        optimizer.step()

        # out = torch.squeeze(predict.detach().cpu())
        # pred = out > 0.5
        # correct += (pred == label_data).sum()
        ### using NLL or CrossEntropyLoss
        pred = predict.max(dim=-1)[-1]
        correct += pred.eq(label_data).sum().item()
        total += len(label_data)
        train_loss += loss.item()

    ### test ###
    model.eval()
    with torch.no_grad():
        for val_batch_id, val_data in enumerate(train_vec_loader(batch_size=128, mode='test', input=X, target=label)()):
            val_x, val_y = val_data
            val_x = val_x.to(device)
            val_y = val_y.to(device)

            val_predict = model(val_x)
            val_pred = val_predict.max(dim=-1)[-1]
            val_correct += val_pred.eq(val_y).sum().item()

            val_total += len(val_y)
            val_loss += criterion(val_predict, val_y.long()).item()

    train_loss_list.append(train_loss / total)
    training_acc.append(int(correct) / total * 100)
    val_loss_list.append(val_loss / val_total)
    testing_acc.append(int(val_correct) / val_total * 100)

    if epoch % 50 == 0:
        print(f"====>Training: Epoch: {epoch}, Train loss: {train_loss_list[-1]:.3f}, Accuracy: {training_acc[-1]:.3f}")
        print(f"Test loss: {val_loss_list[-1]:.3f}, Accuracy: {testing_acc[-1]:.3f}")
        # print(f"Epoch: {epoch}, Loss: {running_loss/total}")

history = {
    "train_loss": train_loss_list,
    "train_acc": training_acc,
    "test_loss": val_loss_list,
    "test_acc": testing_acc,
}
history = pd.DataFrame(history)

#########################################################
# %% Plot result
#########################################################
plot_train_result(history, save_path=None)