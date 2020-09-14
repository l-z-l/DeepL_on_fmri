import torch
from torch import nn
from torch.nn import functional as F
from models.layers import GraphConv
from utils.config import args

class GCN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()

        self.input_dim = input_dim  # 1433
        self.output_dim = output_dim

        print('input dim:', input_dim)
        print('output dim:', output_dim)

        self.layers = nn.Sequential(GraphConv(self.input_dim, args.hidden,
                                                     activation=F.relu,
                                                     is_sparse_inputs=False),
                                    GraphConv(args.hidden, output_dim,
                                                     activation=F.relu,
                                                     dropout=0.4,
                                                     is_sparse_inputs=False),

                                    )

    def forward(self, inputs):
        x, support = inputs
        x = self.layers((x, support))

        return x

    def l2_loss(self):
        layer = self.layers.children()
        layer = next(iter(layer))

        loss = None

        for p in layer.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()

        return loss

if __name__ == "__main__":
    net = GNN(11, len(classes))