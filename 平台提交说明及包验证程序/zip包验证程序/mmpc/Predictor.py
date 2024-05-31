import os
from .model import *
import numpy as np
import random

class Predictor():
    def __init__(self):
        input_size = 5
        hidden_size = 64
        num_layers = 1
        num_classes = 3
        self.model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dicts = torch.load("./mmpc/best_val_model_lstm_label_5.pth", map_location=self.device) ## 注意这里要确保路径能被加载, 使用./model/[my_model_file]路径
        self.model.load_state_dict(state_dicts)  ## 加载模型参数，此处应和训练时的保存方式一致
        self.model.to(self.device)
        self.model.eval()

    def predict(self,x):
        with torch.no_grad():
            x = self.preprocess(x)                  # DataFrame
            x = np.ascontiguousarray(x.values)      # np [sample_num=100, feature_num=5]
            x = torch.tensor(x).to(torch.float32).unsqueeze(0).to(self.device)
            y = self.model(x)                       # shape:[1, output]
            y = y.argmax(dim=1).cpu().numpy()
            return list(y)
    
    def preprocess(self, df):
        ''' 数据预处理 '''        
        df['bsize1'] = df['n_bsize1'].apply(lambda x: np.log1p(x * 100000))
        df['asize1'] = df['n_asize1'].apply(lambda x: np.log1p(x * 100000))
        df['n_bid1'] = df['n_bid1'].apply(lambda x: x * 100)
        df['n_ask1'] = df['n_ask1'].apply(lambda x: x * 100)
        df['n_midprice'] = df['n_midprice'].apply(lambda x: x * 100)
        feature_col_names = ['n_midprice', 'n_bid1', 'bsize1', 'n_ask1', 'asize1']
        return df[feature_col_names]