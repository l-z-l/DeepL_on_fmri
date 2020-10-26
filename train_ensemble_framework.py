from datetime import datetime

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import torch.utils.data as data_utils
from utils.data import *
from models.MLP import Linear
from models.ensemble import Ensemblers
# datasets
from utils.datasets import ConnectivityDatasets, DatasetFactory
from torch.utils.data import DataLoader
# plot
import matplotlib.pyplot as plt

from utils.helper import num_correct, plot_train_result

##########################################################
# %% Meta
###############train_test_split###########################
SAVE = True
MODEL_NANE = f'EnsembleMLP_{datetime.now().strftime("%Y-%m-%d-%H:%M")}'
datadir = './data'
outdir = './outputs'
dataset_name = '273_Havard_Oxford_MSDL_ICA'
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

if SAVE:
    save_path = os.path.join(outdir, f'{MODEL_NANE}_{dataset_name}/') if SAVE else ''
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
else:
    save_path = ''

##########################################################
# %% load data
###############train_test_split###########################
device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
BATCH_SIZE = 64
# load the data
# train_dataset = ConnectivityDatasets('data/augmented/',
#                                roi_type=["train_273_Havard_Oxford_org_100_window_5_stride",
#                                          "train_273_ICA_200_n50_org_100_window_5_stride",
#                                          "train_273_MSDL_org_100_window_5_stride"],
#                                num_subject=2088)
# test_dataset = ConnectivityDatasets('data/augmented/',
#                                roi_type=["test_273_Havard_Oxford_org_100_window_5_stride",
#                                          "test_273_ICA_200_n50_org_100_window_5_stride",
#                                          "test_273_MSDL_org_100_window_5_stride"],
#                                num_subject=369)
train_dataset = ConnectivityDatasets('./data',
                               roi_type=["Havard_Oxford_train",
                                         "MSDL_train",
                                         "ICA_200_n50_train"],
                               num_subject=273)
test_dataset = ConnectivityDatasets('./data',
                               roi_type=["Havard_Oxford_test",
                                         "MSDL_test",
                                         "ICA_200_n50_test"],
                               num_subject=273)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# initialise models
models = []
for s in train_dataset.shape:
    model = Linear(s, 1)
    model.to(device)
    models.append(model)
ensembler = Ensemblers(models, output_dim=2, model_out_dim=1, num_models=len(models))
ensembler.to(device)

# %%
optimizer = optim.Adam(ensembler.parameters(), lr=1e-1, weight_decay=5e-2)
train_loss_list, test_loss_list, training_acc, testing_acc = [], [], [], []
criterion = torch.nn.CrossEntropyLoss().to(device)

for epoch in range(1000):
    train_loss, correct, total = 0, 0, 0
    val_loss, val_correct, val_total = 0, 0, 0
    for batch_id, (data, target) in enumerate(train_loader):
        data = [d.to(device) for d in data]
        target = target.to(device)

        # Feedforward
        # print(list(ensembler.parameters()))
        # print("=" * 20)
        output = ensembler(data)
        loss = criterion(output.squeeze(), target.long())

        # back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += num_correct(output, target)
        total += len(target)
        train_loss += loss.item()

        train_loss_list.append(train_loss / total)
        training_acc.append(int(correct) / total * 100)

        ### validate
        model.eval()
        with torch.no_grad():
            for val_batch_id, (val_x, val_y) in enumerate(train_loader):
                val_x = [d.to(device) for d in val_x]
                val_y = val_y.to(device)

                val_predict = ensembler(val_x)
                val_correct += num_correct(val_predict, val_y)

                val_total += len(val_y)
                val_loss += criterion(val_predict.squeeze(), val_y.long()).item()

        test_loss_list.append(val_loss / val_total)
        testing_acc.append(int(val_correct) / val_total * 100)

    if epoch % 100 == 0:
        print(
            f"====>Training: Epoch: {epoch}, Train loss: {train_loss_list[-1]:.3f}, Accuracy: {training_acc[-1]:.3f}")
        print(f"Test loss: {test_loss_list[-1]:.3f}, Accuracy: {testing_acc[-1]:.3f}")

history = {
    "train_loss": train_loss_list,
    "train_acc": training_acc,
    "test_loss": test_loss_list,
    "test_acc": testing_acc,
}
history = pd.DataFrame(history)

#########################################################
# %% Plot result
#########################################################
plot_train_result(history, save_path=save_path)