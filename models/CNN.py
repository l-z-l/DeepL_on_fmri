import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.layers import FactorizedConvolution


class SpatialTemporalCNN(nn.Module):
    """

    """

    def __init__(self, input_dim=1, output_dim=2):
        super(SpatialTemporalCNN, self).__init__()
        # self.l1 = nn.MaxPool2d((2, 1),1)  # max_pool on ROI level, no need for that, we are not images
        self.l1 = FactorizedConvolution(input_dim, channel_dim=16, bottle_neck=False)
        self.l2 = FactorizedConvolution(16, 32, 8)
        self.l3 = FactorizedConvolution(32, 32, 8)
        self.l4 = FactorizedConvolution(32, 64, 16)
        self.l5 = FactorizedConvolution(64, 64, 16)
        # self.l6 = FactorizedConvolution(64,128,32)
        # self.l7 = FactorizedConvolution(128,128,32)
        self.final_layer = nn.Linear(64, output_dim)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x, return_avg=False):
        """
        input x: shape (bs, 1, 90, 16)  (batch_size, channel, num_ROI, time_step)
        """
        h1 = self.l1(x)
        pool1 = self.maxpool(h1)
        h2 = self.l2(pool1)
        h3 = self.l3(h2)
        pool2 = self.maxpool(h3)
        h4 = self.l4(pool2)
        h5 = self.l5(h4)
        # pool3 = self.maxpool(h5)
        # h6 = self.l6(pool3)
        # h7 = self.l7(h6) # (bs,128, 11,2)
        # print("h7 shape", h7.shape)
        avg_pool = torch.mean(h5, [2, 3])  # (bs, 128)
        # print("avg shape", avg_pool.shape)
        output = self.final_layer(avg_pool)
        return (output, avg_pool) if (return_avg == True) else output
