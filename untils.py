import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
import torch.nn.functional as F


def load_data(file_path):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
    dataset = pd.read_csv(file_path, usecols=[0,1],parse_dates=['date'], index_col='date', date_parser=dateparse)
    dataset.dropna(axis=0, how='any', inplace=True)
    dataset.index.name = 'date'
    return dataset

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def normalize_and_make_series(dataset, look_back):
    values = dataset.values
    values = values.astype('float64')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    column_num = dataset.columns.size
    reframed = series_to_supervised(scaled, look_back, 1)
    # drop columns we don't want to predict
    drop_column = []
    for i in range(look_back * column_num+1, (look_back + 1) * column_num):
        drop_column.append(i)
    reframed.drop(reframed.columns[drop_column], axis=1, inplace=True)
    return reframed, scaler

def split_data(dataset, reframed, look_back, split_time):
    column_num = dataset.columns.size
    train_size = len(dataset[dataset.index < split_time])

    values = reframed.values
    train = values[:train_size, :]
    test = values[train_size:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape(train_X.shape[0], look_back, column_num)
    test_X = test_X.reshape(test_X.shape[0], look_back, column_num)
    train_X = torch.DoubleTensor(train_X)
    train_y = torch.DoubleTensor(train_y)
    test_X = torch.DoubleTensor(test_X)
    test_y = torch.DoubleTensor(test_y)
    return train_X, train_y, test_X, test_y