import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import dataset

# from models.layers import GraphConv
from utils.config import args
from utils.data import list_2_tensor

import torch.nn as nn
from models.layers import AvgReadout

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.nn import GraphConv, SAGPooling, JumpingKnowledge


class GNN_SAG(torch.nn.Module):
    def __init__(self, num_features, nhid, num_classes=2, pooling_ratio=0.5, dropout_ratio=0.3):
        super(GNN_SAG, self).__init__()
        self.num_features = num_features
        self.nhid = nhid
        self.num_classes = num_classes
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout_ratio

        self.conv1 = GraphConv(self.num_features, self.nhid)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = GraphConv(self.nhid, self.nhid)
        self.pool2 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GraphConv(self.nhid, self.nhid)
        self.pool3 = SAGPooling(self.nhid, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        # x = F.log_softmax(self.lin3(x), dim=-1)

        return self.lin3(x)


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(GNN, self).__init__()
        # GraphConv Applied neighbourhood normalization
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)


        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GCN(nn.Module):

    def __init__(self, input_dim, output_dim, node_num=116):
        super(GCN, self).__init__()

        self.input_dim = input_dim  # 11
        self.output_dim = output_dim  # 2

        print('input dim:', input_dim)
        print('output dim:', output_dim)
        self.layers = nn.Sequential(GraphConv(self.input_dim, hidden,
                                              activation=F.relu,
                                              is_sparse_inputs=True),
                                    GraphConv(hidden, output_dim * 2,
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
