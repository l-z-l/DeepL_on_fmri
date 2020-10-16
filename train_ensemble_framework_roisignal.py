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
from models.LSTM import LSTM
# datasets
from utils.datasets import *
from torch.utils.data import DataLoader
# plot
import matplotlib.pyplot as plt

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

# load the data
train_datasets = ["2088_train_273_Havard_Oxford_org_100_window_5_stride",
                  "2088_train_273_ICA_200_n50_org_100_window_5_stride",
                  "2088_train_273_MSDL_org_100_window_5_stride"]
test_datasets = ["369_test_273_Havard_Oxford_org_100_window_5_stride",
                 "369_test_273_ICA_200_n50_org_100_window_5_stride",
                 "369_test_273_MSDL_org_100_window_5_stride"]
test_datasets = RoiSignalDatasets(dataDir='data/augmented', datasets=test_datasets)
train_datasets = RoiSignalDatasets(dataDir='data/augmented', datasets=train_datasets)
train_loader = DataLoader(train_datasets, batch_size=64, shuffle=True)
test_loader = DataLoader(test_datasets, batch_size=64, shuffle=True)
# initialise models
models = []
for s in train_datasets[0][0]:
    model = LSTM(input_size=s.shape[1], hidden_dim=16, seq_len=s.shape[0], num_layers=1, output_size=1)
    model.to(device)
    models.append(model)
ensemble_layer = Ensemblers(1, 1, len(models))
ensemble_layer.to(device)

for epoch in range(2):
    total = 0
    correct = 0
    for batch_id, (data, target) in enumerate(train_loader):
        # some magic code to produce the output that only @Joey can understand
        output = [models[i](data[i]) for i in range(len(data))]

        output = ensemble_layer(torch.reshape(torch.cat(output), (1, len(models))))
        # delete from here
        loss = F.binary_cross_entropy(output, target)
        pred = (output >= 0.5).float()
        correct += (pred == target).float().sum()
        total += target.size()[0]
        accuracy = 100 * correct / total
        print('ep:%5d loss: %6.4f acc: %5.2f' %
              (epoch, loss.item(), accuracy))
