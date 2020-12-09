from datetime import datetime

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader

from utils.data import *
from models.CNN import SpatialTemporalCNN
from utils.config import args
from utils.datasets import DatasetFactory
from utils.helper import train_loader, train_loader_graph, plot_train_result, num_correct

from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
# plot
import matplotlib.pyplot as plt

##########################################################
# %% Meta
###############train_test_split###########################
SAVE = False
MODEL_NANE = f'2dCNN_{datetime.now().strftime("%Y-%m-%d-%H:%M")}'
datadir = './data/augmented/'
outdir = './outputs'
dataset_name = '271_AAL_org_100_window_5_stride'
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

if SAVE:
    save_path = os.path.join(outdir, f'{MODEL_NANE}_{dataset_name}/') if SAVE else ''
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
else:
    save_path = ''

##########################################################
# %% Load Data
##########################################################
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
## augmented
# train_dataset, test_dataset = DatasetFactory.create_train_test_roi_signal_datasets_from_path(
#     train_path="data/augmented/2088_train_273_ICA_200_n50_org_100_window_5_stride",
#     test_path="data/augmented/369_test_273_ICA_200_n50_org_100_window_5_stride")
## not augmented
train_dataset, test_dataset = DatasetFactory.create_train_test_roi_signal_datasets_from_single_path(
    path="./data/273_Havard_Oxford"
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

##########################################################
# %% initialise mode and
##########################################################
model = SpatialTemporalCNN()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
criterion = torch.nn.CrossEntropyLoss().to(device)

train_loss_list, val_loss_list, training_acc, testing_acc = [], [], [], []
for epoch in range(500):
    model.train()
    train_loss, correct, total = 0, 0, 0
    val_loss, val_correct, val_total = 0, 0, 0

    ### train ###
    for batch_id, (train_x, train_y) in enumerate(train_loader):
        # Preparing Data
        train_x = torch.unsqueeze(train_x, 1).to(device)  # add extra dimension (m, 1, time_seq, ROIS)
        train_y = train_y.to(device)

        # Feedforward
        optimizer.zero_grad()
        predict = model(train_x)
        # Compute the loss
        loss = criterion(predict, train_y.long())
        loss.backward()
        optimizer.step()

        correct += num_correct(predict, train_y)
        total += len(train_y)
        train_loss += loss.item()

    train_loss_list.append(train_loss / total)
    training_acc.append(int(correct) / total * 100)

    ### test ###
    model.eval()
    with torch.no_grad():
        for val_batch_id, (val_x, val_y) in enumerate(test_loader):
            val_x = torch.unsqueeze(val_x, 1).to(device)
            val_y = val_y.to(device)

            val_predict = model(val_x)
            val_correct += num_correct(val_predict, val_y)

            val_total += len(val_y)
            val_loss += criterion(val_predict, val_y.long()).item()

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
plot_train_result(history, save_path=save_path)