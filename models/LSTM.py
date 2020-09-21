import torch
from torch import nn
from torch.nn import functional as F
from models.layers import GraphConv
from utils.config import args


class LSTM(nn.Module):
    def __init__(self, input_size=functional_connectivities.shape[1], hidden_layer_size=100, num_layers=10,
                 output_size=2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(num_layers, 1, self.hidden_layer_size),
                            torch.zeros(num_layers, 1, self.hidden_layer_size))

        # self.log_softmax = nn.LogSoftmax(dim=1)
        self.sig = nn.Sigmoid()

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        print(f"Predictions shape {predictions.shape}")
        predictions = self.sig(predictions)
        print(f"predictions shape {type(predictions)}")
        output, inds = torch.max(predictions, dim=1)
        # output.requires_grad=True
        # sourceTensor.clone()
        # output = torch.tensor(output)
        output = torch.tensor(output, dtype=torch.float)
        print(f"output shape {type(output)}")
        return output