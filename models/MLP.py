import torch
from torch import nn


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
            nn.Linear(input_dim, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        ### weight initialisation
        for m in self.modules():
            if type(m) == nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.)

    def forward(self, x):
        return self.MLP(x)  # Each element i,j is a scalar in R. f(xi,proj_j)
