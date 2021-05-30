import torch
import torch.nn as nn
import torch.nn.functional as F

class FastText(torch.nn.Module):

    def __init__(self,embed_num, embed_dim, class_num, hidden_size, dropout=0.5):
        super(FastText,self).__init__()

        self.embed = nn.Embedding(embed_num, embed_dim)
        self.embed_bigram = nn.Embedding(embed_num, embed_dim)
        self.embed_trigram = nn.Embedding(embed_num, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, class_num)

    def forward(self, x):
        embed_cbow = self.embed(x[0])
        embed_bigram = self.embed(x[1])
        embed_trigram = self.embed(x[2])

        out = embed_cbow + embed_bigram + embed_trigram
        out = out.mean(dim=1)

        logit = self.dropout(out)
        logit = self.fc(logit)
        return logit
