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
        self.model_5 = LSTMModel(input_size, hidden_size, num_layers, num_classes)
        self.model_10 = LSTMModel(input_size, hidden_size, num_layers, num_classes)
        self.model_20 = LSTMModel(input_size, hidden_size, num_layers, num_classes)
        self.model_40 = LSTMModel(input_size, hidden_size, num_layers, num_classes)
        self.model_60 = LSTMModel(input_size, hidden_size, num_layers, num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dicts_5 = torch.load("./mmpc/best_val_model_lstm_label_5.pth", map_location=self.device) ## 注意这里要确保路径能被加载, 使用./model/[my_model_file]路径
        self.model_5.load_state_dict(state_dicts_5)  ## 加载模型参数，此处应和训练时的保存方式一致
        self.model_5.to(self.device)
        self.model_5.eval()

        state_dicts_10 = torch.load("./mmpc/best_val_model_lstm_label_10.pth",map_location=self.device)
        self.model_10.load_state_dict(state_dicts_10)
        self.model_10.to(self.device)
        self.model_10.eval()

        state_dicts_20 = torch.load("./mmpc/best_val_model_lstm_label_20.pth",map_location=self.device)
        self.model_20.load_state_dict(state_dicts_20)
        self.model_20.to(self.device)
        self.model_20.eval()

        state_dicts_40 = torch.load("./mmpc/best_val_model_lstm_label_40.pth",map_location=self.device)
        self.model_40.load_state_dict(state_dicts_40)
        self.model_40.to(self.device)
        self.model_40.eval()

        state_dicts_60 = torch.load("./mmpc/best_val_model_lstm_label_60.pth",map_location=self.device)
        self.model_60.load_state_dict(state_dicts_60)
        self.model_60.to(self.device)
        self.model_60.eval()

    def predict(self,x):
        with torch.no_grad():
            x = self.preprocess(x)                  # DataFrame
            x = np.ascontiguousarray(x.values)      # np [sample_num=100, feature_num=5]
            x = torch.tensor(x).to(torch.float32).unsqueeze(0).to(self.device)
            y_5 = self.model_5(x)                       # shape:[1, output]
            y_5 = y_5.argmax(dim=1).cpu().numpy()
            y_10 = self.model_10(x)
            y_10 = y_10.argmax(dim=1).cpu().numpy()
            y_20 = self.model_20(x)
            y_20 = y_20.argmax(dim=1).cpu().numpy()
            y_40 = self.model_40(x)
            y_40 = y_40.argmax(dim=1).cpu().numpy()
            y_60 = self.model_60(x)
            y_60 = y_60.argmax(dim=1).cpu().numpy()
            # return a list of int, len=5
            return [y_5[0], y_10[0], y_20[0], y_40[0], y_60[0]]
    
    def preprocess(self, df):
        ''' 数据预处理 '''        
        df['bsize1'] = df['n_bsize1'].apply(lambda x: np.log1p(x * 100000))
        df['asize1'] = df['n_asize1'].apply(lambda x: np.log1p(x * 100000))
        df['n_bid1'] = df['n_bid1'].apply(lambda x: x * 100)
        df['n_ask1'] = df['n_ask1'].apply(lambda x: x * 100)
        df['n_midprice'] = df['n_midprice'].apply(lambda x: x * 100)
        feature_col_names = ['n_midprice', 'n_bid1', 'bsize1', 'n_ask1', 'asize1']
        return df[feature_col_names]