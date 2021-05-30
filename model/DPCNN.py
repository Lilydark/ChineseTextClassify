import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, embed_num, embed_dim, class_num, kernel_num, dropout=0.5):
        super(Model, self).__init__()

        self.embedding = nn.Embedding(embed_num, embed_dim)
        self.conv_region = nn.Conv2d(1, kernel_num, (3, embed_dim), stride=1)
        self.conv = nn.Conv2d(kernel_num, kernel_num, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(kernel_num, class_num)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.conv_region(x)

        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x