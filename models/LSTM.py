import torch
from torch import nn
from torch.nn import functional as F
from models.layers import GraphConv
from utils.config import args


class LSTM(nn.Module):
    def __init__(self, input_size=48, hidden_layer_size=32, num_layers=1,
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
            nn.Conv1d(hidden_layer_size, hidden_layer_size, kernel_size=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(10)
            # nn.BatchNorm1d(hidden_layer_size)
        )
        self.conv1 = nn.Conv1d(hidden_layer_size,hidden_layer_size, kernel_size=2, padding=3)
        self.max = nn.MaxPool1d(10)

        self.sig = nn.Sigmoid()

    def forward(self, input_seq):
        ### 2 layer LSTM
        lstm_out, (h, c) = self.lstm_1(input_seq)
        lstm_out_1, (h, c) = self.lstm_2(lstm_out)
        output = lstm_out_1.reshape(len(input_seq), -1)
        output = self.linear(output)
        output = self.sig(output)
        # output, _ = torch.max(lstm_out, 1)
        ##### CNN LSTM
        # lstm_out, (h, c) = self.lstm_1(input_seq)
        # lstm_out = lstm_out.permute(0, 2, 1)
        # output = self.CONV(lstm_out)
        # output = torch.max(output, 2)[0]  # global max for CNN
        # output = self.linear(output)
        # output = self.sig (output)

        # Global Max pooling


        return output


    # Potential issue with VAE
        # Can't learn the representation to be classified
        # Current analysis strategy
        #    ignore the temporal information
        #   Try temporal functional connectivity
        #   K-means clustering
        #     50 cluster
        #     50 connectivity temporal matrix
        #     100 fc-temporal connectivity as the input
    # Try ADHD dataset
    # Check the confusion matrix of SVM
    # Find paper on ADNI dataset
    # Try artificial data having distinct difference
        # Try adding dictinctive feature/noise to the data/functional connectivity to see if the DL model work