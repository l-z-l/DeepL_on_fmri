import torch
from torch import nn
from torch.nn import functional as F
from models.layers import GraphConv
from utils.config import args


class LSTM(nn.Module):
    def __init__(self, input_size=116, hidden_layer_size=140, num_layers=1,
                 output_size=2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        # self.lstm = tnn.LSTM(embedding_dim,hidden_dim,n_layer,dropout=drop_prob,batch_first=True,bidirectional=True,bias=True)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        # self.hidden_cell = (torch.zeros(num_layers, 1, self.hidden_layer_size),
        #                     torch.zeros(num_layers, 1, self.hidden_layer_size))

        # self.log_softmax = nn.LogSoftmax(dim=1)
        self.sig = nn.Sigmoid()

    def forward(self, input_seq):
        lstm_out, (h, c) = self.lstm(input)
        # lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        # Batch size
        output, _ = torch.max(lstm_out, 1)
        output = self.linear(output)
        print(f"lstm_out shape {lstm_out.shape}")
        print(f"output shape {output.shape}")

        return output