import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    def __init__(self,
                 config: dict,
                 num_classes):
        super(CNN, self).__init__()

        print('allocate convolution  layers')
        filter_width = config['filter_size']
        kernels = config['kernels']
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=config['lfcc_size'],
                      out_channels=filter_width,
                      kernel_size=kernel)
            for kernel in kernels
        ])

        self.fc = nn.Linear(filter_width * len(kernels), num_classes)
        self.dropout = nn.Dropout(p=config['dropout'])

    def forward(self, signals):
        # signals = [#batch size, lfcc dim, speech length]

        x_conv_list = [F.relu(conv(signals)) for conv in self.convs]
        # x_conv = [batch size, out_channel_width, speech length - kernel size + 1]
        
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list]
        # x_pool = [batch size, out_channel_width, 1]
        
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)
        # x_fc = [batch size, out_channel_width * #kernels]
        
        logits = self.fc(self.dropout(x_fc))
        # logits = [batch size, #classes]
        
        return logits
