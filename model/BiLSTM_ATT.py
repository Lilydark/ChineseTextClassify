import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class BiLSTM_Attention(torch.nn.Module):
    def __init__(self, embed_num, embed_dim, class_num, hidden_size, bidirectional, cuda, attention_size, dropout=0.5):
        super(BiLSTM_Attention, self).__init__()

        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.class_num = class_num
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.attention_size = attention_size
        self.dropout = dropout
        self.cuda = cuda

        self.embed = nn.Embedding(embed_num, embed_dim)
        self.embed.weight.data.uniform_(-1, 1)

        if bidirectional:
            self.layer_size = 2
        else:
            self.layer_size = 1

        self.lstm = nn.LSTM(embed_dim, hidden_size, 1, dropout=dropout, bidirectional=bidirectional)

        self.w_omega = Variable(torch.zeros(hidden_size * self.layer_size, attention_size))
        self.u_omega = Variable(torch.zeros(attention_size))

        if self.cuda:
            self.w_omega = self.w_omega.cuda()
            self.u_omega = self.u_omega.cuda()

        self.fc = nn.Linear(hidden_size * self.layer_size, class_num)

    def attention(self, lstm_output):
        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size*self.layer_size])
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length, 1])
        state = lstm_output.permute(1, 0, 2)
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output

    def forward(self, x):
        x = self.embed(x).permute(1, 0, 2)
        self.sequence_length, self.batch_size = x.size()[:2]

        h_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))
        c_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))

        if self.cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(x, (h_0, c_0))
        attn_output = self.attention(lstm_output)
        logits = self.fc(attn_output)

        return logits
