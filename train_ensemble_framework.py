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

from utils.helper import num_correct

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
train_dataset1, test_dataset1 = DatasetFactory.create_train_test_connectivity_datasets_from_single_path(
    path="data/273_MSDL"
)
train_dataset2, test_dataset2 = DatasetFactory.create_train_test_connectivity_datasets_from_single_path(
    path="data/273_Havard_Oxford"
)
train_dataset3, test_dataset3 = DatasetFactory.create_train_test_connectivity_datasets_from_single_path(
    path="data/273_ICA_200_n50"
)
train_dataset = ConnectivityDatasets(datasets=[train_dataset1, train_dataset2, train_dataset3])
test_dataset = ConnectivityDatasets(datasets=[test_dataset1, test_dataset2, test_dataset3])

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
optimizer = optim.Adam(ensembler.parameters(), lr=1e-3, weight_decay=5e-3)
train_loss_list, test_loss_list, training_acc, testing_acc = [], [], [], []
criterion = torch.nn.CrossEntropyLoss().to(device)

for epoch in range(5):
    train_loss, correct, total = 0, 0, 0
    val_loss, val_correct, val_total = 0, 0, 0
    for batch_id, (data, target) in enumerate(train_loader):
        data = [d.to(device) for d in data]
        target = target.to(device)

        # Feedforward
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

    # if epoch % 10 == 0:
    print(
        f"====>Training: Epoch: {epoch}, Train loss: {train_loss_list[-1]:.3f}, Accuracy: {training_acc[-1]:.3f}")
    print(f"Test loss: {test_loss_list[-1]:.3f}, Accuracy: {testing_acc[-1]:.3f}")
