import os
import numpy as np
import torch

from datasets.utils.mtrics_PEMS08D4 import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler, Add_Window_Horizon


def get_adjacency_matrix(distance_df_filename, num_of_vertices, type='distance'):
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        # header = f.__next__()
        # print(reader)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), int(float(row[2]))
            if type == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            elif type == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            else:
                raise ValueError('Adj type error!!!!!')
    
    return [A]


def load_st_dataset(dataset, file):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join(file)
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join(file)
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data


def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def generate_seq_dataset_PEMS(dataset='PEMSD4', file='data/PeMSD4/pems04.npz', normalizer='std', column_wise=False):

    data = load_st_dataset(dataset=dataset, file=file)

    data, scaler = normalize_dataset(data, normalizer, column_wise)
    # scaler = None

    n_tasks = 5

    num_all = data.shape[0]
    num_samples = int(num_all/3)
    num_test = int((num_all-num_samples)/n_tasks)

    train_list = []
    val_list = []
    test_list = []

    for i in range(n_tasks):
        if i == 0:
            data_flag = data[:num_samples]
            train_len = int(len(data_flag)*0.8)
            data_train = data_flag[:train_len]
            data_val = data_flag[train_len:]
            data_test = data[num_samples:num_samples+num_test]

            x_train, y_train = Add_Window_Horizon(data_train, 12, 12, single=False)
            x_val, y_val = Add_Window_Horizon(data_val, 12, 12, single=False)
            x_test, y_test = Add_Window_Horizon(data_test, 12, 12, single=False)
            train_list.append((x_train, y_train))
            val_list.append((x_val, y_val))
            test_list.append((x_test, y_test))

            # print('Train: ', x_train.shape, y_train.shape)
            # print('Val: ', x_val.shape, y_val.shape)
            # print('Test: ', x_test.shape, y_test.shape)
        else:
            data_flag = data[num_samples+(i-1)*num_test:num_samples+i*num_test]
            train_len = int(len(data_flag)*0.8)
            data_train = data_flag[:train_len]
            data_val = data_flag[train_len:]
            data_test = data[num_samples+i*num_test:num_samples+(i+1)*num_test]

            x_train, y_train = Add_Window_Horizon(data_train, 12, 12, single=False)
            x_val, y_val = Add_Window_Horizon(data_val, 12, 12, single=False)
            x_test, y_test = Add_Window_Horizon(data_test, 12, 12, single=False)
            train_list.append((x_train, y_train))
            val_list.append((x_val, y_val))
            test_list.append((x_test, y_test))

            # print('Train: ', x_train.shape, y_train.shape)
            # print('Val: ', x_val.shape, y_val.shape)
            # print('Test: ', x_test.shape, y_test.shape)
    
    return train_list, val_list, test_list, scaler