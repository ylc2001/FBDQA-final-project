import os
from .model import *
import numpy as np

class Predictor():
    def __init__(self):
        self.model = deeplob(num_classes = 3)   
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dicts = torch.load("./mmpc/best_val_model.pth", map_location=self.device) ## 注意这里要确保路径能被加载, 使用./model/[my_model_file]路径
        self.model.load_state_dict(state_dicts)  ## 加载模型参数，此处应和训练时的保存方式一致
        self.model.to(self.device)
        self.model.eval()

    def predict(self,x):
        with torch.no_grad():
            x = self.preprocess(x)
            x = torch.tensor(x.values).to(torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            y = self.model(x) # shape:[1, output]
            y = y.cpu().numpy()
            y = self.generate_signal(y, single_label=True) # shape:[1, label_num]
            y = list(y.reshape(-1)) # shape:[label_num]
            return y
    
    ## 这里不重要，预处理方式因人而异
    def preprocess(self, df):
        ''' 数据预处理 '''        
        df['n_bid1'] = df['n_bid1']+1
        df['n_bid2'] = df['n_bid2']+1
        df['n_bid3'] = df['n_bid3']+1
        df['n_bid4'] = df['n_bid4']+1
        df['n_bid5'] = df['n_bid5']+1
        df['n_ask1'] = df['n_ask1']+1
        df['n_ask2'] = df['n_ask2']+1
        df['n_ask3'] = df['n_ask3']+1
        df['n_ask4'] = df['n_ask4']+1
        df['n_ask5'] = df['n_ask5']+1

        # 量价组合
        df['spread1'] =  df['n_ask1'] - df['n_bid1']
        df['spread2'] =  df['n_ask2'] - df['n_bid2']
        df['spread3'] =  df['n_ask3'] - df['n_bid3']
        df['mid_price1'] =  df['n_ask1'] + df['n_bid1']
        df['mid_price2'] =  df['n_ask2'] + df['n_bid2']
        df['mid_price3'] =  df['n_ask3'] + df['n_bid3']
        df['weighted_ab1'] = (df['n_ask1'] * df['n_bsize1'] + df['n_bid1'] * df['n_asize1']) / (df['n_bsize1'] + df['n_asize1'])
        df['weighted_ab2'] = (df['n_ask2'] * df['n_bsize2'] + df['n_bid2'] * df['n_asize2']) / (df['n_bsize2'] + df['n_asize2'])
        df['weighted_ab3'] = (df['n_ask3'] * df['n_bsize3'] + df['n_bid3'] * df['n_asize3']) / (df['n_bsize3'] + df['n_asize3'])

        df['vol1_rel_diff']   = (df['n_bsize1'] - df['n_asize1']) / (df['n_bsize1'] + df['n_asize1'])
        df['volall_rel_diff'] = (df['n_bsize1'] + df['n_bsize2'] + df['n_bsize3'] + df['n_bsize4'] + df['n_bsize5'] \
                        - df['n_asize1'] - df['n_asize2'] - df['n_asize3'] - df['n_asize4'] - df['n_asize5'] ) / \
                        ( df['n_bsize1'] + df['n_bsize2'] + df['n_bsize3'] + df['n_bsize4'] + df['n_bsize5'] \
                        + df['n_asize1'] + df['n_asize2'] + df['n_asize3'] + df['n_asize4'] + df['n_asize5'] )

        df['amount'] = df['amount_delta'].map(np.log1p)
        feature_col_names = ['n_bid1','n_bsize1',
                     'n_bid2','n_bsize2',
                     'n_bid3','n_bsize3',
                     'n_bid4','n_bsize4',
                     'n_bid5','n_bsize5',
                     'n_ask1','n_asize1',
                     'n_ask2','n_asize2',
                     'n_ask3','n_asize3',
                     'n_ask4','n_asize4',
                     'n_ask5','n_asize5',
                     'spread1','mid_price1',
                     'spread2','mid_price2',
                     'spread3','mid_price3',
                     'weighted_ab1','weighted_ab2','weighted_ab3','amount',
                     'vol1_rel_diff','volall_rel_diff'
                    ]
        return df[feature_col_names]
    
    def generate_signal(self, predict_matrix, class_num=3, single_label=True):
        '''
        Args:
            predict_matrix: np [sample_num, class_num * label_num] if single_label = False
                            np [sample_num, class_num] if single_label = True
        Returns:
            signal: np [sample_num, label_num] if single_label = False
                    np [sample_num] if single_label = True
        '''
        if single_label:
            signal = predict_matrix.argmax(axis=1) # shape:[sample_num]
            return signal
        else:
            signal =  predict_matrix.reshape(predict_matrix.shape[0], class_num, predict_matrix.shape[1] // class_num)
            signal = signal.transpose(0,2,1) # shape:[sample_num, label_num, class_num]
            signal = signal.argmax(axis=2) # shape:[sample_num, label_num]
            return signal

