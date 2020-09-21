import torch
from torch import nn
from torch.nn import functional as F
from models.layers import GraphConv
from utils.config import args


class LSTM(nn.Module):
    def __init__(self, input_size=116, hidden_layer_size=32, num_layers=1,
                 output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm_1 = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.lstm_2 = nn.LSTM(hidden_layer_size, 2, num_layers, batch_first=True)
        # self.lstm = tnn.LSTM(embedding_dim,hidden_dim,n_layer,dropout=drop_prob,batch_first=True,bidirectional=True,bias=True)

        self.linear = nn.Linear(280, output_size)

        # self.hidden_cell = (torch.zeros(num_layers, 1, self.hidden_layer_size),
        #                     torch.zeros(num_layers, 1, self.hidden_layer_size))

        # self.log_softmax = nn.LogSoftmax(dim=1)
        self.CONV = nn.Sequential(
            nn.Conv1d(hidden_layer_size, hidden_layer_size, kernel_size=3, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.BatchNorm1d(hidden_layer_size)
        )

        self.sig = nn.Sigmoid()

    def forward(self, input_seq):
        lstm_out, (h, c) = self.lstm_1(input_seq)
        lstm_out_1, (h, c) = self.lstm_2(lstm_out)
        # lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        # Batch size
        # output, _ = torch.max(lstm_out, 1)
        print(f"lstm_out shape {lstm_out.shape}")
        print(f"lstm_out_1 shape {lstm_out_1.shape}")
        # lstm_out = lstm_out.permute(0, 2, 1)
        # output = self.CONV(lstm_out)
        # print(f"output shape {output.shape}")
        output = []
        output = lstm_out_1.reshape(len(input_seq), -1)
        print(f"output shape {output.shape}")
        output = self.linear(output)
        output = self.sig(output)
        # print(f"lstm_out2 shape {lstm_out.shape}")

        return output