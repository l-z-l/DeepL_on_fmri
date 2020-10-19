from torch import nn
from torch.nn import functional as F
import torch

class Ensemblers(nn.Module):
    """
        Ensemble layer
    """

    def __init__(self, models, output_dim=1, model_out_dim=2, num_models=1):
        super(Ensemblers, self).__init__()
        assert models != None

        self.voters = nn.ModuleList(models)
        self.fc = nn.Sequential(
            nn.Tanh(),
            nn.Linear(num_models * model_out_dim, output_dim)
        )

    def forward(self, x):
        assert len(x) == 3

        x = torch.stack([self.voters[i](data) for i, data in enumerate(x)])
        # print(x.shape)
        x = x.view(x.shape[1], x.shape[0])
        # print(x.shape)
        ### voting ###
        x = self.fc(x)
        return x