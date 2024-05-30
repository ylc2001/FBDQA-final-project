# %%
import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn.functional as F
from torch.utils import data
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim

# %% [markdown]
# # 读取数据

# %%
file_dir = "/l1/data/FBDQA2021A_MMP_Challenge_ver0.2/data"

sym = 4
dates = list(range(24))
df = pd.DataFrame()
for date in dates:
    if (date & 1):
        file_name = f"snapshot_sym{sym}_date{date//2}_am.csv"
    else:
        file_name = f"snapshot_sym{sym}_date{date//2}_pm.csv"
    if not os.path.isfile(os.path.join(file_dir,file_name)):
        continue
    new_df = pd.read_csv(os.path.join(file_dir,file_name))
    # 价格+1（从涨跌幅还原到对前收盘价的比例）
    new_df['bid1'] = new_df['n_bid1']+1
    new_df['bid2'] = new_df['n_bid2']+1
    new_df['bid3'] = new_df['n_bid3']+1
    new_df['bid4'] = new_df['n_bid4']+1
    new_df['bid5'] = new_df['n_bid5']+1
    new_df['ask1'] = new_df['n_ask1']+1
    new_df['ask2'] = new_df['n_ask2']+1
    new_df['ask3'] = new_df['n_ask3']+1
    new_df['ask4'] = new_df['n_ask4']+1
    new_df['ask5'] = new_df['n_ask5']+1
    
    # 量价组合
    new_df['spread1'] =  new_df['ask1'] - new_df['bid1']
    new_df['spread2'] =  new_df['ask2'] - new_df['bid2']
    new_df['spread3'] =  new_df['ask3'] - new_df['bid3']
    new_df['mid_price1'] =  new_df['ask1'] + new_df['bid1']
    new_df['mid_price2'] =  new_df['ask2'] + new_df['bid2']
    new_df['mid_price3'] =  new_df['ask3'] + new_df['bid3']
    new_df['weighted_ab1'] = (new_df['ask1'] * new_df['n_bsize1'] + new_df['bid1'] * new_df['n_asize1']) / (new_df['n_bsize1'] + new_df['n_asize1'])
    new_df['weighted_ab2'] = (new_df['ask2'] * new_df['n_bsize2'] + new_df['bid2'] * new_df['n_asize2']) / (new_df['n_bsize2'] + new_df['n_asize2'])
    new_df['weighted_ab3'] = (new_df['ask3'] * new_df['n_bsize3'] + new_df['bid3'] * new_df['n_asize3']) / (new_df['n_bsize3'] + new_df['n_asize3'])
    
    new_df['vol1_rel_diff']   = (new_df['n_bsize1'] - new_df['n_asize1']) / (new_df['n_bsize1'] + new_df['n_asize1'])
    new_df['volall_rel_diff'] = (new_df['n_bsize1'] + new_df['n_bsize2'] + new_df['n_bsize3'] + new_df['n_bsize4'] + new_df['n_bsize5'] \
                     - new_df['n_asize1'] - new_df['n_asize2'] - new_df['n_asize3'] - new_df['n_asize4'] - new_df['n_asize5'] ) / \
                     ( new_df['n_bsize1'] + new_df['n_bsize2'] + new_df['n_bsize3'] + new_df['n_bsize4'] + new_df['n_bsize5'] \
                     + new_df['n_asize1'] + new_df['n_asize2'] + new_df['n_asize3'] + new_df['n_asize4'] + new_df['n_asize5'] )
                            
    new_df['amount'] = new_df['amount_delta'].map(np.log1p)
    df = df.append(new_df)
    




# %%
df

# %%
feature_col_names = ['bid1','n_bsize1',
                     'bid2','n_bsize2',
                     'bid3','n_bsize3',
                     'bid4','n_bsize4',
                     'bid5','n_bsize5',
                     'ask1','n_asize1',
                     'ask2','n_asize2',
                     'ask3','n_asize3',
                     'ask4','n_asize4',
                     'ask5','n_asize5',
                     'spread1','mid_price1',
                     'spread2','mid_price2',
                     'spread3','mid_price3',
                     'weighted_ab1','weighted_ab2','weighted_ab3','amount',
                     'vol1_rel_diff','volall_rel_diff'
                    ]
label_col_name = ['label_5']

# %%
n = len(df)
##划分训练/测试集
train_nums = int(n*0.8)
val_nums = int(n*0.1)
print(f'train_nums: {train_nums}, val_nums: {val_nums}, test_nums: {n-train_nums-val_nums}')
train_data = np.ascontiguousarray(df[feature_col_names][:train_nums].values)
train_label = df[label_col_name][:train_nums].values.reshape(-1)

val_data = np.ascontiguousarray(df[feature_col_names][train_nums:train_nums+val_nums].values)
val_label = df[label_col_name][train_nums:train_nums+val_nums].values.reshape(-1)

test_data = np.ascontiguousarray(df[feature_col_names][train_nums+val_nums:].values)
test_label = df[label_col_name][train_nums+val_nums:].values.reshape(-1)

# %% [markdown]
# ## GPU准备

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %% [markdown]
# ## 准备dataset

# %%
def data_transform(X, T):
    [N, D] = X.shape
    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = X[i - T:i, :]
    return dataX

# %%
class Dataset(data.Dataset):
    def __init__(self, data, label,  num_classes, T):
        self.T = T

        data = data_transform(data, self.T)

        self.x = torch.tensor(data).to(torch.float32).unsqueeze(1).to(device)

#         self.y = F.one_hot(torch.tensor(label[T - 1:].astype(np.int64)), num_classes=3)
        self.y = torch.tensor(label[T - 1:].astype(np.int64)).to(device)
    
        self.length = len(self.x)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x[index], self.y[index]

# %%
batch_size = 512

dataset_train = Dataset(data=train_data,label=train_label, num_classes=3, T=100)
dataset_val   = Dataset(data=val_data,  label=val_label,   num_classes=3, T=100)
dataset_test  = Dataset(data=test_data, label=test_label,  num_classes=3, T=100)

train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
val_loader   = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
test_loader  = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

print(dataset_train.x.shape, dataset_train.y.shape, dataset_train.x.requires_grad, dataset_train.y.requires_grad,)
print(dataset_val.x.shape, dataset_val.y.shape, dataset_val.x.requires_grad, dataset_val.y.requires_grad,)
print(dataset_test.x.shape, dataset_test.y.shape, dataset_test.x.requires_grad, dataset_test.y.requires_grad,)

# %%
print(dataset_train.x.device)
print(dataset_train.y.device)

# %% [markdown]
# # 定义模型

# %%
class deeplob(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
#             nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5,1),stride=(2,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1),stride=(2,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,8)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1),stride=(2,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        
        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(5,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
       
        # lstm layers
        self.fc = nn.Sequential(nn.Linear(384, 64),nn.Linear(64, self.num_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        x = x.reshape(-1,48*8)
        x = self.fc(x)

        forecast_y = torch.softmax(x, dim=1)

        return forecast_y

# %%
model = deeplob(num_classes = 3)
model.to(device);

# %%
summary(model, (1, 1, 100, 32))

# %% [markdown]
# # 训练模型

# %%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay = 1e-5)

# %%
def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs):
    
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    best_test_loss = np.inf
    best_test_epoch = 0

    for it in tqdm(range(epochs)):
        if ((epochs+1) % 10 == 0):
            optimizer.lr = optimizer.lr*0.5
        model.train()
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()
            
            optimizer.step()
            
            train_loss.append(loss.item())
            
        # Get train loss and test loss
        train_loss = np.mean(train_loss) # a little misleading
    
        model.eval()
        test_loss = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)      
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss
        
        if test_loss < best_test_loss:
            torch.save(model, f'best_val_model_pytorch_sym{sym}_date{dates[-1]}')
            best_test_loss = test_loss
            best_test_epoch = it
            print('model saved')

        dt = datetime.now() - t0
        print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, \
          Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}')
    torch.save(model, f'final_model_pytorch_sym{sym}_date{dates[-1]}')
    return train_losses, test_losses

# %%
train_losses, val_losses = batch_gd(model, criterion, optimizer, 
                                    train_loader, val_loader, epochs=50)

# %%
plt.figure(figsize=(16,6))
plt.plot(train_losses, label='train loss')
plt.plot(val_losses, label='validation loss')
plt.legend()

# %%


# %% [markdown]
# # 测试模型

# %%
model = torch.load(f'best_val_model_pytorch_sym{sym}_date{dates[-1]}', map_location=device)
model.eval()
all_targets = []
all_predictions = []

for inputs, targets in test_loader:
    # Move to GPU
    inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

    # Forward pass
    outputs = model(inputs)
    
    # Get prediction
    # torch.max returns both max and argmax
    _, predictions = torch.max(outputs, 1)

    all_targets.append(targets.cpu().numpy())
    all_predictions.append(predictions.cpu().numpy())

all_targets = np.concatenate(all_targets)    
all_predictions = np.concatenate(all_predictions)    
print('accuracy_score:', accuracy_score(all_targets, all_predictions))
print(classification_report(all_targets, all_predictions, digits=4))

# %%
# model = torch.load('best_val_model_pytorch',map_location=device)
all_targets = []
all_predictions = []

for inputs, targets in train_loader:

    # Forward pass
    outputs = model(inputs)
    
    # Get prediction
    # torch.max returns both max and argmax
    _, predictions = torch.max(outputs, 1)

    all_targets.append(targets.cpu().numpy())
    all_predictions.append(predictions.cpu().numpy())

all_targets = np.concatenate(all_targets)    
all_predictions = np.concatenate(all_predictions) 
print('accuracy_score:', accuracy_score(all_targets, all_predictions))
print(classification_report(all_targets, all_predictions, digits=4))

# %%


# %% [markdown]
# # 过拟合：小规模数据难以充分训练大模型

# %%
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.net = nn.Sequential(
                    nn.Linear(3200,128),
                    nn.LeakyReLU(),
                    nn.Linear(128,64),
                    nn.LeakyReLU(),
                    nn.Linear(64,64),
                    nn.LeakyReLU(),
                    nn.Linear(64,3)
                )
        
    def forward(self,x):
        x = x.view(-1,100*32)
        x = self.net(x)
        return torch.softmax(x, dim=1)

# %%
model = MLP()
model.to(device)
summary(model,(1,1,100,32))

# %%
criterion = nn.CrossEntropyLoss(weight = torch.Tensor([2,1,2]).to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)#, momentum=0.9, weight_decay = 1e-5)

# %%
train_losses, val_losses = batch_gd(model, criterion, optimizer, 
                                    train_loader, val_loader, epochs=500)

# %%
model = torch.load(f'best_val_model_pytorch_sym{sym}_date{dates[-1]}', map_location=device)

all_targets = []
all_predictions = []

for inputs, targets in test_loader:
    # Move to GPU
    inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

    # Forward pass
    outputs = model(inputs)
    
    # Get prediction
    # torch.max returns both max and argmax
    _, predictions = torch.max(outputs, 1)

    all_targets.append(targets.cpu().numpy())
    all_predictions.append(predictions.cpu().numpy())

all_targets = np.concatenate(all_targets)    
all_predictions = np.concatenate(all_predictions)    
print('accuracy_score:', accuracy_score(all_targets, all_predictions))
print(classification_report(all_targets, all_predictions, digits=4))


