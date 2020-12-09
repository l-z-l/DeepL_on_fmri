import torch
from torch import nn
from torch.nn import functional as F
from models.layers import GraphConv
from utils.config import args


class LSTM(nn.Module):
    def __init__(self, input_size=48, hidden_dim=64, seq_len=140, num_layers=1,
                 output_size=2):
        super().__init__()
        self.hidden_layer_size = hidden_dim

        self.lstm_1 = nn.LSTM(input_size, hidden_dim, num_layers, dropout=0.5, batch_first=True)
        self.lstm_2 = nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=0.5, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_size)
        # self.hidden_cell = (torch.zeros(num_layers, 1, self.hidden_layer_size),
        #                     torch.zeros(num_layers, 1, self.hidden_layer_size))

        self.CONV = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=seq_len//10),
            nn.Dropout(0.5)
            # nn.BatchNorm1d(hidden_layer_size)
        )

    def interpret(self, input_seq):
        ##### CNN LSTM
        lstm_out, (h, c) = self.lstm_1(input_seq)
        lstm_out, _ = self.lstm_2(lstm_out)
        lstm_out = lstm_out.permute(0, 2, 1)
        output = self.CONV(lstm_out)
        output = torch.max(output, 2)[0]  # global max for CNN
        return output

    def forward(self, input_seq):
        '''
        ### 2 layer LSTM
        # lstm_out, (h, c) = self.lstm_1(input_seq)
        lstm_out, _ = self.lstm_1(input_seq)
        lstm_out_1, _ = self.lstm_2(lstm_out)
        # lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        # Batch size
        # output, _ = torch.max(lstm_out, 1)
        # print(f"lstm_out shape {lstm_out.shape}")
        # print(f"lstm_out_1 shape {lstm_out_1.shape}")
        # lstm_out = lstm_out.permute(0, 2, 1)
        # output = self.CONV(lstm_out)
        # print(f"output shape {output.shape}")
        output = lstm_out_1.reshape(len(input_seq), -1)
        output = self.linear(output)
        # output = self.sig(output)
        # output, _ = torch.max(lstm_out, 1)
        '''
        ##### CNN LSTM
        lstm_out, (h, c) = self.lstm_1(input_seq)
        lstm_out, _ = self.lstm_2(lstm_out)
        lstm_out = lstm_out.permute(0, 2, 1)
        output = self.CONV(lstm_out)
        output = torch.max(output, 2)[0]  # global max for CNN
        output = self.linear(output)

        ### Global Max pooling
        # output = self.sig(output)
        # print(f"lstm_out2 shape {lstm_out.shape}")

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