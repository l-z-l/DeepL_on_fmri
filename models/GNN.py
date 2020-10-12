import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import dataset

from utils.config import args
from utils.data import list_2_tensor

import torch.nn as nn
from models.layers import AvgReadout

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.nn import GraphConv, SAGPooling, JumpingKnowledge, BatchNorm, HypergraphConv, GATConv


class Hyper_GCN(torch.nn.Module):
    def __init__(self, num_features, nhid, num_classes=2, pooling_ratio=0.5, dropout_ratio=0.3):
        super(Hyper_GCN, self).__init__()
        self.num_features = num_features
        self.nhid = nhid
        self.num_classes = num_classes
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout_ratio
        self.multiheads = 1

        ###
        # 1st layer
        self.conv1 = HypergraphConv(self.num_features, self.nhid, use_attention=False, heads=self.multiheads, dropout=0)
        self.pool1 = SAGPooling(self.nhid * self.multiheads, ratio=self.pooling_ratio)
        # 2nd layer
        self.conv2 = HypergraphConv(self.nhid, self.nhid, use_attention=False, heads=self.multiheads, dropout=0)
        self.pool2 = SAGPooling(self.nhid * self.multiheads, ratio=self.pooling_ratio)
        # 3rd layer
        self.conv3 = HypergraphConv(self.nhid, self.nhid, use_attention=False, heads=self.multiheads, dropout=0)
        self.pool3 = SAGPooling(self.nhid * self.multiheads, ratio=self.pooling_ratio)
        # fc layer
        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        # 1st layer
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        # 2nd layer
        x = F.relu(self.conv2(x, edge_index))
        # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        # 3rd layer
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        # concat
        x = x1 + x2 + x3
        # fully connected MLP
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # x = F.log_softmax(self.lin3(x), dim=-1)

        return self.lin3(x)

class GNN_SAG(torch.nn.Module):
    def __init__(self, num_features, nhid, num_classes=2, pooling_ratio=0.5, dropout_ratio=0.3):
        super(GNN_SAG, self).__init__()
        self.num_features = num_features
        self.nhid = nhid
        self.num_classes = num_classes
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout_ratio

        ###
        # 1st layer
        self.conv1 = GraphConv(self.num_features, self.nhid)
        self.bn1 = nn.BatchNorm1d(self.nhid)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        # 2nd layer
        self.conv2 = GraphConv(self.nhid, self.nhid)
        self.bn2 = nn.BatchNorm1d(self.nhid)
        self.pool2 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        # 3rd layer
        self.conv3 = GraphConv(self.nhid, self.nhid)
        self.bn3 = nn.BatchNorm1d(self.nhid)
        self.pool3 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        # fc layer
        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        # 1st layer
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        # 2nd layer
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        # 3rd layer
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        # concat
        x = x1 + x2 + x3
        # fully connected MLP
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
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
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin(x)

        return x

if __name__ == "__main__":
    pass
