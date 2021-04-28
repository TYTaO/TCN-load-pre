import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
import torch.nn.functional as F
import argparse
from untils import *
from model import TCN
import time

parser = argparse.ArgumentParser(description='Sequence Modeling - load_pre Task')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clip, -1 means no clip (default: 1.0)')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit (default: 50)')
parser.add_argument('--ksize', type=int, default=2,
                    help='kernel size (default: 8)')
parser.add_argument('--iters', type=int, default=100,
                    help='number of iters per epoch (default: 100)')
parser.add_argument('--levels', type=int, default=3,
                    help='# of levels (default: 8)')
# parser.add_argument('--blank_len', type=int, default=1000, metavar='N',
#                     help='The size of the blank (i.e. T) (default: 1000)')
parser.add_argument('--seq_len', type=int, default=64,
                    help='initial history size (default: 64)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval (default: 50')
parser.add_argument('--lr', type=float, default=5e-5,
                    help='initial learning rate (default: 5e-4)')
parser.add_argument('--optim', type=str, default='RMSprop',
                    help='optimizer to use (default: RMSprop)')
parser.add_argument('--nhid', type=int, default=64,
                    help='number of hidden units per layer (default: 10)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
args = parser.parse_args()

torch.manual_seed(args.seed)

file_path = 'D:\Study\实验室\TCN\Fisher-model-master\container1.csv'
split_time = pd.datetime.strptime('2018-08-14 00:00:00', '%Y-%m-%d %H:%M:%S') # 数据集范围 2018-07-24 00:00:00 ~ 2018-08-24 00:00:00
look_back = 512
batch_size = args.batch_size
seq_len = args.seq_len    # The size to memorize
epochs = args.epochs
iters = args.iters
n_classes = 1  # out size
n_train = 10000
n_test = 1000
model_para_path = './model/02-512'

dataset = load_data(file_path)
# scaler限制到0-1
reframed, scaler = normalize_and_make_series(dataset, look_back)
train_x, train_y, test_x, test_y = split_data(dataset, reframed, look_back, split_time)

channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
dropout = args.dropout
model = TCN(look_back, n_classes, channel_sizes, kernel_size, dropout=dropout)
model.double()

if args.cuda:
    model.cuda()
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    test_x = test_x.cuda()
    test_y = test_y.cuda()

criterion = nn.MSELoss()
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def evaluate():
    model.eval()
    test_size = 50 # GPU内存不足
    show_size = 10 # show
    with torch.no_grad():
        x, y = test_x[0:test_size], test_y[0:test_size]
        output = model(x)
        output = torch.squeeze(output)
        test_loss = F.mse_loss(output, y)
        print('\nTest set: Average loss: {:.6f}\n'.format(test_loss.item()))
        print(y[0:show_size])
        print(output[0:show_size])
        return test_loss.item()


def train(epoch):
    global lr
    model.train()
    batch_idx = 1
    total_loss = 0

    for i in range(0, train_x.size(0), batch_size):
        if i + batch_size > train_x.size(0):
            x, y = train_x[i:], train_y[i:]
        else:
            x, y = train_x[i:(i+batch_size)], train_y[i:(i+batch_size)]
        optimizer.zero_grad()
        output = model(x)
        output = torch.squeeze(output)
        loss = F.mse_loss(output, y)
        loss.backward()
    #     if args.clip > 0:
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        batch_idx += 1
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            cur_loss = total_loss / 100
            processed = min(i+batch_size, train_x.size(0))
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, train_x.size(0), 100.*processed/train_x.size(0), lr, cur_loss))
            total_loss = 0


for ep in range(1, epochs + 1):
    train(ep)
    torch.save(model.state_dict(), model_para_path)
    evaluate()



