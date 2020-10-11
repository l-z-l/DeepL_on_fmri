import torch, math
from torch import nn
from torch.nn import functional as F
from utils.helper import sparse_dropout, dot
from utils.config import args
from torch.nn.init import xavier_normal_
from utils.data import list_2_tensor
import numpy as np


class Ensemblers(nn.Module):
    """
        Ensemble layer
    """

    def __init__(self, output_dim=1, model_out_dim=2, num_models=1):
        super(Ensemblers, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_models * model_out_dim, output_dim),
            nn.ReLU()

        )

    def forward(self, x):
        x = self.net(x)
        return x
