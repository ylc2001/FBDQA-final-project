import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.notebook import tqdm 
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn.functional as F
from torch.utils import data
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim

def data_transform(X, T):
    [N, D] = X.shape
    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = X[i - T:i, :]
    return dataX

feature_col_names = ['n_midprice', 'n_bid1', 'bsize1', 'n_ask1', 'asize1']
label_col_name = 'label_5'

file_dir = "D:\my-dev-code\FBDQA-final-project\FBDQA2021A_MMP_Challenge_ver0.2\data"
os.makedirs('./np_data', exist_ok=True)

# sym = 0 - 9
sym_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
dates = list(range(79))
ampm = ['am', 'pm']
nparray_all_data = np.array([])
nparray_all_label = np.array([])
for sym in sym_list:
    for date in dates:
        df_day = pd.DataFrame()
        for _ampm in ampm:
            if (_ampm == 'am'):
                file_name = f"snapshot_sym{sym}_date{date}_am.csv"
            else:
                file_name = f"snapshot_sym{sym}_date{date}_pm.csv"
            if not os.path.isfile(os.path.join(file_dir,file_name)):
                continue
            new_df = pd.read_csv(os.path.join(file_dir,file_name))
            print("Processing", file_name)
            # 价格+1（从涨跌幅还原到对前收盘价的比例）
            new_df['bsize1'] = new_df['n_bsize1'].map(lambda x: np.log1p(x * 100000))
            new_df['asize1'] = new_df['n_asize1'].map(lambda x: np.log1p(x * 100000))
            new_df['n_bid1'] = new_df['n_bid1'].map(lambda x: x * 100)
            new_df['n_ask1'] = new_df['n_ask1'].map(lambda x: x * 100)
            new_df['n_midprice'] = new_df['n_midprice'].map(lambda x: x * 100)
            df_day = pd.concat([df_day, new_df])
        try:
            nparray_day_data = np.ascontiguousarray(df_day[feature_col_names].values)
            nparray_day_data = data_transform(nparray_day_data, 100)   # [N_days, T=100, D_features]
            nparray_day_label = df_day[label_col_name].values.reshape(-1)
            nparray_day_label = nparray_day_label[99:]  # [N_days,]
            if nparray_all_data.size == 0:
                nparray_all_data = nparray_day_data
            else:
                nparray_all_data = np.vstack((nparray_all_data, nparray_day_data))
            if nparray_all_label.size == 0:
                nparray_all_label = nparray_day_label
            else:
                nparray_all_label = np.hstack((nparray_all_label, nparray_day_label))
            # print("Data shape: ", nparray_day_data.shape)
            # print("Label shape: ", nparray_day_label.shape)
        except:
            print("Error in sym:", sym, "date:", date)
            print("Columns in df_day: ", df_day.columns)
            print("Columns in feature_col_names: ", feature_col_names)
            print("Data type of df_day columns: ", df_day.columns.dtype)
            print("Data type of feature_col_names: ", type(feature_col_names[0]))
            continue

print("All data shape: ", nparray_all_data.shape)
print("All label shape: ", nparray_all_label.shape)

# save to file
with open(f'./np_data/nparray_all_data.pkl', 'wb') as f:
    pickle.dump(nparray_all_data, f)
with open(f'./np_data/nparray_all_label_{label_col_name}.pkl', 'wb') as f:
    pickle.dump(nparray_all_label, f)