import torch, math
from torch import nn
from torch.nn import functional as F
from utils.helper import sparse_dropout, dot
from utils.config import args
from torch.nn.init import xavier_normal_
from utils.data import list_2_tensor


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)



class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim,
                 dropout=0.,
                 is_sparse_inputs=False,
                 activation=F.relu,
                 featureless=False,
                 init="xavier"):
        super(GraphConv, self).__init__()

        self.dropout = dropout
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))

        ### weight initialisation
        if init == 'xavier':
            print("-" * 10, "> Xavier Initialization")
            xavier_normal_(self.weight.data,
                           gain=0.02)  # gain=math.sqrt(2. / (1 + 0.01)))  # Gain adapted for LeakyReLU activation function
            self.bias.data.fill_(0.01)
        elif init == 'random':
            print("-" * 10, "> Rnadom Initialization")
            # Todo:
        elif init == 'uniform':
            print("-" * 10, "> Uniform Initialization")
            # Todo:
        elif init == 'kaiming':
            print("-" * 10, "> Kaiming Initialization")
            nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
            self.bias.data.fill_(0.01)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        x, adj = inputs

        if self.training:
            x = F.dropout(x, self.dropout)

        # convolve
        out_list = []
        for i, mx in enumerate(x):
            if not self.featureless:  # if it has features x
                if self.is_sparse_inputs:
                    # print()
                    xw = torch.sparse.mm(mx, self.weight)
                else:
                    xw = torch.mm(mx, self.weight)  # (20, 116, 16) (116, 2)  -> (20, 116, 2)
            else:
                # initial pass
                xw = self.weight

            out = torch.sparse.mm(adj[i], xw)  # (116, 116)  (20, 116, out_dim) -> (20, 116, out_dim)
            out_list.append(out)

        # if self.bias is not None:
        #     out += self.bias

        out_list = list_2_tensor(out_list)

        return self.activation(out_list), adj


class FactorizedConvolution(nn.Module):
    """
    4 layers,
    1x1, 1x3, 3x1, 1x1, which reduces half of the direct convolution parameters
    """

    def __init__(self, input_dim=1, output_dim=1, channel_dim=32, bottle_neck=True):
        super(FactorizedConvolution, self).__init__()
        layers = [nn.Conv2d(input_dim, channel_dim, 1, 1),  # 1*1*1*1
                  nn.ReLU(),
                  nn.Conv2d(channel_dim, channel_dim, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                  # padding = (0, 1) to remain original shape
                  nn.ReLU(),
                  nn.Conv2d(channel_dim, channel_dim, kernel_size=(3, 1), stride=1, padding=(1, 0)),
                  nn.ReLU(),
                  nn.Conv2d(channel_dim, output_dim, 1, 1)
                  ]
        if bottle_neck == False:
            layers = [nn.Conv2d(input_dim, channel_dim, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                      nn.ReLU(),
                      nn.Conv2d(channel_dim, channel_dim, kernel_size=(3, 1), stride=1, padding=(1, 0))
                      ]
        self.identity_match_layer = nn.Conv2d(input_dim, output_dim, 1, 1)
        self.layers = nn.Sequential(*layers)
        self.relu = nn.ReLU()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        hidden = self.layers(x)
        if self.input_dim != self.output_dim:
            x = self.identity_match_layer(x)
        x = self.relu(x + hidden)  # residual
        return x


class GraphAttention(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(in_features, out_features).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        f_1 = torch.matmul(h, self.a1)
        f_2 = torch.matmul(h, self.a2)
        e = self.leakyrelu(f_1 + f_2.transpose(0, 1))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
