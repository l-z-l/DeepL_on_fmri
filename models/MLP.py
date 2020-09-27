import torch, math
from torch import nn
from torch.nn import functional as F
from utils.helper import sparse_dropout, dot
from utils.config import args
from torch.nn.init import xavier_normal_
from utils.data import list_2_tensor


class Linear(nn.Module):
    '''MLP, that take in input both input data and latent codes and output
    an unique value in R.
    This network defines the function t(x,z) that appears in the variational
    representation of a f-divergence. Finding t() that maximize this f-divergence,
    lead to a variation representation that is a tight bound estimator of mutual information.
    '''

    def __init__(self, input_dim, output_dim=2):
        super(Linear, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # self.linear = nn.Linear(input_dim, output_dim)

        self.MLP = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1000, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        ### weight initialisation
        for m in self.modules():
            if type(m) == nn.Linear:
                torch.nn.init.normal_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        return torch.sigmoid(self.MLP(x))  # Each element i,j is a scalar in R. f(xi,proj_j)
