import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCnn(nn.Module):
    def __init__(self, embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout = 0.5):
        super(TextCnn, self).__init__()

        Ci = 1
        Co = kernel_num

        self.embed = nn.Embedding(embed_num, embed_dim) # 全连接层，学习词向量
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (f, embed_dim), padding = (2, 0)) for f in kernel_sizes]) # 卷积层，Ci和Co为进出的通道数量，padding就是卷积的padding，kernel_sizes卷积核大小，准备多个窗口大小不同的卷积
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(Co * len(kernel_sizes), class_num) # 全连接层

    def forward(self, x):
        x = self.embed(x)  # (N, token_num, embed_dim)
        x = x.unsqueeze(1)  # (N, Ci, token_num, embed_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, token_num) * len(kernel_sizes)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co) * len(kernel_sizes)]
        x = torch.cat(x, 1) # (N, Co * len(kernel_sizes))
        x = self.dropout(x)  # (N, Co * len(kernel_sizes))
        logit = self.fc(x)  # (N, class_num)
        return logit