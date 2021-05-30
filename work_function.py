import torch
import torch.nn.functional as F
import os
import pandas
import jieba
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import re

def train(trainloader, validloader, testloader, epochs, model, device, optimizer):
    model.to(device)
    steps = 0
    min_valid_loss = float('inf')

    for epoch in range(1, epochs + 1):
        print ('==============Epoch: ', epoch, '/', epochs, '==============')
        avg_loss_train, avg_loss_valid = 0, 0
        for batch, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logit = model(x)
            loss = F.cross_entropy(logit, y)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % 20 == 0:
                acc = (sum(logit.argmax(axis=1) == y) / len(y) * 100).item()
                acc_percent = str(round(acc, 2)) + '%'
                print ('Step:', steps, 'Loss:', round(loss.item(), 6), 'acc:',acc_percent)

        valid_acc = 0
        for batch, (x, y) in enumerate(validloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logit = model(x)
            loss = F.cross_entropy(logit, y)
            avg_loss_valid += loss.item()
            acc = (sum(logit.argmax(axis=1) == y) / len(y) * 100).item()
            valid_acc += acc
        valid_acc /= len(validloader)

        train_acc = 0
        for batch, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logit = model(x)
            loss = F.cross_entropy(logit, y)
            avg_loss_train += loss.item()
            acc = (sum(logit.argmax(axis=1) == y) / len(y) * 100).item()
            train_acc += acc
        train_acc /= len(trainloader)

        print('train_loss:', round(avg_loss_train / len(trainloader), 6), 'train_acc:', str(round(train_acc, 2))+'%')
        print('valid_loss:', round(avg_loss_valid / len(validloader), 6), 'valid_acc:', str(round(valid_acc, 2))+'%')

        if avg_loss_valid < min_valid_loss:
            torch.save(model.state_dict(), 'output/temp.pkl')
            min_valid_loss = avg_loss_valid

    model.load_state_dict(torch.load('output/temp.pkl'))

    avg_loss_test = 0; test_acc = 0
    for batch, (x, y) in enumerate(testloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logit = model(x)
        loss = F.cross_entropy(logit, y)
        avg_loss_test += loss.item()
        acc = (sum(logit.argmax(axis=1) == y) / len(y) * 100).item()
        test_acc += acc

    test_acc /= len(testloader)

    print ()
    print('Final test_loss:', round(avg_loss_test / len(testloader), 6), 'test_acc:', str(round(test_acc, 2))+'%')

    os.remove('output/temp.pkl')
    return model

def data2vec(name, batch_size=64, max_len=200): # 适用于读取文本自行生成本地编码词典的情况
    train = pd.read_csv('data/' + name + '_train.csv', encoding='utf8')
    test = pd.read_csv('data/' + name + '_test.csv', encoding='utf8')
    valid = pd.read_csv('data/' + name + '_valid.csv', encoding='utf8')

    data_list = [train, valid, test]
    for data in data_list:
        data['text'] = data.text.fillna('').apply(lambda x: jieba.lcut(x))
    # 获取词典
    words = set()
    for data in [train]:
        set_series = data.text.apply(lambda x: set(x))
        for i in range(len(set_series)):
            words = words | set_series[i]
    word_dic = {}
    for word in words:
        word_dic[word] = len(word_dic)

    def wordbag(text, word_dic=word_dic, max_len=max_len):
        if len(text) < max_len:
            text = text + (max_len - len(text)) * [len(word_dic) + 1]
        return [word_dic.get(x, len(word_dic)) for x in text][:max_len]

    for data in data_list:
        data['wordvec'] = data.text.fillna('').apply(lambda x: wordbag(x))

    dataloader_list = []
    for data in data_list:
        dataset_i = TensorDataset(torch.tensor(data.wordvec), torch.tensor(data.label))
        data_loader = DataLoader(dataset=dataset_i, batch_size=batch_size)
        dataloader_list.append(data_loader)
        del data_loader
    trainloader, validloader, testloader = dataloader_list
    tokenizer_size = len(word_dic) + 2
    return trainloader, validloader, testloader, tokenizer_size

# 拆分文本处理流程：准备一个函数是把单组文本返回单组词向量与tokenizer的
def data2vector(name, cut_method, max_len=200):

    def ngram(word, n):
        return [word[x:x + n] for x in range(len(word) - n + 1)]

    def jieba_cut(word):
        return jieba.lcut(word)

    train = pd.read_csv('data/' + name + '_train.csv', encoding='utf8')
    test = pd.read_csv('data/' + name + '_test.csv', encoding='utf8')
    valid = pd.read_csv('data/' + name + '_valid.csv', encoding='utf8')
    data_list = [train, valid, test]

    if cut_method == 'jieba':
        for data in data_list:
            data['text'] = data.text.fillna('').apply(lambda x: jieba.lcut(x))
    elif re.match('[123456789]gram', cut_method):
        for data in data_list:
            data['text'] = data.text.fillna('').apply(lambda x: ngram(x, int(cut_method[0])))

    # 获取词典
    words = set()
    for data in [train]:
        set_series = data.text.apply(lambda x: set(x))
        for i in range(len(set_series)):
            words = words | set_series[i]
    word_dic = {}
    for word in words:
        word_dic[word] = len(word_dic)

    def wordbag(text, word_dic=word_dic, max_len=max_len):
        if len(text) < max_len:
            text = text + (max_len - len(text)) * [len(word_dic) + 1]
        return [word_dic.get(x, len(word_dic)) for x in text][:max_len]

    for data in data_list:
        data['wordvec'] = data.text.fillna('').apply(lambda x: wordbag(x))

    train_wordvec, valid_wordvec, test_wordvec = train['wordvec'], valid['wordvec'], test['wordvec']
    train_y, valid_y, test_y, = train.label, valid.label, test.label
    tokenizer_size = len(word_dic) + 2
    return train_wordvec, valid_wordvec, test_wordvec, tokenizer_size, train_y, valid_y, test_y

def data2loader4fastText(name, cut_method_list=['jieba', '2gram', '3gram'], batch_size=64, max_len=200):
    train_wordvec_list, valid_wordvec_list, test_wordvec_list = [], [], []
    for cut_method in cut_method_list:
        train_wordvec, valid_wordvec, test_wordvec, tokenizer_size, train_y, valid_y, test_y = data2vector(name, cut_method, max_len)
        train_wordvec_list.append(train_wordvec)
        valid_wordvec_list.append(valid_wordvec)
        test_wordvec_list.append(test_wordvec)

    train_wordvec_list.append(train_y)
    valid_wordvec_list.append(valid_y)
    test_wordvec_list.append(test_y)

    wordvec_lists = [train_wordvec_list, valid_wordvec_list, test_wordvec_list]

    dataloader_list = []
    for wordvecs in wordvec_lists:
        dataset_i = TensorDataset(torch.tensor(wordvecs[0]), torch.tensor(wordvecs[1]), torch.tensor(wordvecs[2]), torch.tensor(wordvecs[3]))
        data_loader = DataLoader(dataset=dataset_i, batch_size=batch_size)
        dataloader_list.append(data_loader)
        del data_loader


    trainloader, validloader, testloader = dataloader_list
    return trainloader, validloader, testloader, tokenizer_size
def train4fastText(trainloader, validloader, testloader, epochs, model, device, optimizer):
    model.to(device)
    steps = 0
    min_valid_loss = float('inf')

    for epoch in range(1, epochs + 1):
        print ('==============Epoch: ', epoch, '/', epochs, '==============')
        avg_loss_train, avg_loss_valid = 0, 0
        for batch, (x1, x2, x3, y) in enumerate(trainloader):
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
            x = [x1, x2, x3]
            optimizer.zero_grad()
            logit = model(x)
            loss = F.cross_entropy(logit, y)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % 20 == 0:
                acc = (sum(logit.argmax(axis=1) == y) / len(y) * 100).item()
                acc_percent = str(round(acc, 2)) + '%'
                print ('Step:', steps, 'Loss:', round(loss.item(), 6), 'acc:',acc_percent)

        valid_acc = 0
        for batch, (x1, x2, x3, y) in enumerate(validloader):
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
            x = [x1, x2, x3]
            optimizer.zero_grad()
            logit = model(x)
            loss = F.cross_entropy(logit, y)
            avg_loss_valid += loss.item()
            acc = (sum(logit.argmax(axis=1) == y) / len(y) * 100).item()
            valid_acc += acc
        valid_acc /= len(validloader)

        train_acc = 0
        for batch, (x1, x2, x3, y) in enumerate(trainloader):
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
            x = [x1, x2, x3]
            optimizer.zero_grad()
            logit = model(x)
            loss = F.cross_entropy(logit, y)
            avg_loss_train += loss.item()
            acc = (sum(logit.argmax(axis=1) == y) / len(y) * 100).item()
            train_acc += acc
        train_acc /= len(trainloader)

        print('train_loss:', round(avg_loss_train / len(trainloader), 6), 'train_acc:', str(round(train_acc, 2))+'%')
        print('valid_loss:', round(avg_loss_valid / len(validloader), 6), 'valid_acc:', str(round(valid_acc, 2))+'%')

        if avg_loss_valid < min_valid_loss:
            torch.save(model.state_dict(), 'output/temp.pkl')
            min_valid_loss = avg_loss_valid

    model.load_state_dict(torch.load('output/temp.pkl'))

    avg_loss_test = 0; test_acc = 0
    for batch, (x1, x2, x3, y) in enumerate(testloader):
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        x = [x1, x2, x3]
        optimizer.zero_grad()
        logit = model(x)
        loss = F.cross_entropy(logit, y)
        avg_loss_test += loss.item()
        acc = (sum(logit.argmax(axis=1) == y) / len(y) * 100).item()
        test_acc += acc

    test_acc /= len(testloader)

    print ()
    print('Final test_loss:', round(avg_loss_test / len(testloader), 6), 'test_acc:', str(round(test_acc, 2))+'%')

    os.remove('output/temp.pkl')
    return model
