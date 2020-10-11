import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

from utils.data import *
from models.MLP import Linear
from models.ensemble import Ensemblers
from utils.config import args
from utils.helper import train_vec_loader

from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
# plot
import matplotlib.pyplot as plt

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

# data loader
ROIs, labels, labels_index = load_ensembled_data(dataDir='data/',
                                                 roi_type=["Havard_Oxford",
                                                           "ICA_200_n50",
                                                           "MSDL_label",
                                                           "Ward_160n50"])

X_list = []
for item in ROIs:
    X_list.append(
        torch.as_tensor(
            signal_to_connectivities(item,
                                     kind='tangent',
                                     discard_diagonal=True,
                                     vectorize=True)))
labels = [x if (x == "CN") else "CD" for x in labels]
classes, labels_index, classes_count = np.unique(labels, return_inverse=True, return_counts=True)
label = torch.as_tensor(labels_index, dtype=torch.float)

# initialise models
models = []
for X in X_list:
    model = Linear(X.shape[1], 1)
    model.to(device)
    models.append(model)
ensembler = Ensemblers(1, 1, len(models))
for epoch in range(15000):
    X = torch.cat([models[i](X_list[i]) for i in range(0, len(X_list))], dim=0)
    ensembler(X)
