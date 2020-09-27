import torch
from torch import nn
from torch.nn import functional as F
from models.layers import GraphConv
from utils.config import args
from utils.data import list_2_tensor

import torch.nn as nn
from models.layers import GraphConv, AvgReadout



class GCN(nn.Module):

    def __init__(self, input_dim, output_dim, node_num=116):
        super(GCN, self).__init__()

        self.input_dim = input_dim  # 11
        self.output_dim = output_dim # 2

        print('input dim:', input_dim)
        print('output dim:', output_dim)
        self.layers = nn.Sequential(GraphConv(self.input_dim, args.hidden,
                                              activation=F.relu,
                                              is_sparse_inputs=True),
                                    GraphConv(args.hidden, output_dim * 2,
                                              activation=F.relu,
                                              dropout=0.5,
                                              is_sparse_inputs=False),
                                    )
        self.readout = nn.Sequential(nn.Linear(node_num * output_dim * 2, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, output_dim),
                                     nn.LogSoftmax()
                                     )

        ### TODO: weight initilization

    def forward(self, inputs):
        x, support = inputs
        x, _ = self.layers((x, support))

        x = torch.flatten(x, start_dim=1)
        return self.readout(x)

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

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = GraphConv(n_in, n_h, activation=activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1, _ = self.gcn((seq1, adj))

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2, _ = self.gcn((seq2, adj))

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            if isinstance(m, nn.Bilinear):
                torch.nn.init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.contiguous().expand_as(h_pl)

        sc1_list, sc2_list = [], []
        for i, x in enumerate(h_pl):
            z = self.f_k(h_pl[i], c_x[i])
            sc_1 = torch.squeeze(self.f_k(h_pl[i], c_x[i]), -1)
            sc_2 = torch.squeeze(self.f_k(h_mi[i], c_x[i]), -1)
            sc1_list.append(sc_1)
            sc2_list.append(sc_2)

        sc_1, sc_2 = list_2_tensor(sc1_list), list_2_tensor(sc2_list)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

if __name__ == "__main__":
    net = GCN(11, 2)
