from torch import nn


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
