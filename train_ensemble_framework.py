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
from utils.datasets import ConnectivityDatasets
from torch.utils.data import DataLoader
# plot
import matplotlib.pyplot as plt

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

# load the data
roi_types = ["Havard_Oxford",
             "ICA_200_n50",
             "MSDL"]
dataset = ConnectivityDatasets('data/',
                               roi_type=roi_types,
                               num_subject=273)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

# initialise models
models = []
for s in dataset.shape:
    model = Linear(s, 1)
    model.to(device)
    models.append(model)
ensembler = Ensemblers(1, 1, len(models))
ensembler.to(device)

for epoch in range(2):
    total = 0
    correct = 0
    for batch_id, (data, target) in enumerate(train_loader):
        output = [models[i](data[i]) for i in range(len(data))]
        output = ensembler(torch.reshape(torch.cat(output), (1, 3)))
        loss = F.binary_cross_entropy(output, target)
        pred = (output >= 0.5).float()
        correct += (pred == target).float().sum()
        total += target.size()[0]
        accuracy = 100 * correct / total
        print('ep:%5d loss: %6.4f acc: %5.2f' %
              (epoch, loss.item(), accuracy))
