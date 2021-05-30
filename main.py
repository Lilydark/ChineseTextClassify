from work_function import *
from model import TextCNN
from model import BiLSTM_ATT
from model import FastText
from model import DPCNN

import torch
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True,)
args = parser.parse_args()
print ('Model:',args.model)


with open('parameter/' + args.model + '.json','r') as f:
    parameter = json.loads(f.readline())
    print ('parameter:',parameter)

if args.model == 'TextCNN':
    trainloader, validloader, testloader, tokenizer_size = data2vec('ChnSentiCorp', parameter['batch_size'], parameter['max_len'])
    model = TextCNN.TextCnn(embed_num=tokenizer_size, embed_dim=parameter['embed_dim'], class_num=parameter['class_num'], kernel_num=parameter['kernel_num'], kernel_sizes=parameter['kernel_sizes'])
    model = train(trainloader, validloader, testloader, epochs=parameter['epochs'], model=model, device=torch.device(parameter['device']),optimizer=torch.optim.Adam(model.parameters(), lr=parameter['lr']))

if args.model == 'DPCNN':
    trainloader, validloader, testloader, tokenizer_size = data2vec('ChnSentiCorp', parameter['batch_size'], parameter['max_len'])
    model = DPCNN.Model(embed_num=tokenizer_size, embed_dim=parameter['embed_dim'], class_num=parameter['class_num'], kernel_num=parameter['kernel_num'])
    model = train(trainloader, validloader, testloader, epochs=parameter['epochs'], model=model, device=torch.device(parameter['device']), optimizer=torch.optim.Adam(model.parameters(), lr=parameter['lr']))

if args.model == 'LSTM_Attention':
    trainloader, validloader, testloader, tokenizer_size = data2vec('ChnSentiCorp', parameter['batch_size'], parameter['max_len'])
    model = BiLSTM_ATT.BiLSTM_Attention(embed_num=tokenizer_size, embed_dim=parameter['embed_dim'], class_num=parameter['class_num'], hidden_size=parameter['hidden_size'], cuda=parameter['cuda'], bidirectional=parameter['bidirectional'], attention_size=parameter['attention_size'])
    model = train(trainloader, validloader, testloader, epochs=parameter['epochs'], model=model, device=torch.device(parameter['device']),optimizer=torch.optim.Adam(model.parameters(), lr=parameter['lr']))

if args.model == 'FastText':
    trainloader, validloader, testloader, tokenizer_size = data2loader4fastText('ChnSentiCorp',['jieba', '2gram', '3gram'], parameter['batch_size'], parameter['max_len'])
    model = FastText.FastText(embed_num=tokenizer_size, embed_dim=parameter['embed_dim'], class_num=parameter['class_num'], hidden_size=parameter['hidden_size'])
    model = train4fastText(trainloader, validloader, testloader, epochs=parameter['epochs'], model=model, device=torch.device(parameter['device']), optimizer=torch.optim.Adam(model.parameters(), lr=parameter['lr']))


torch.save(model, 'output/model.pkl')

