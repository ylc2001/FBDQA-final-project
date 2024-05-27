import sys
import pandas as pd
import json

# 定义动态导入类函数
def import_class(import_str):
    mod_str,_seq,class_str = import_str.rpartition(".")
    # 导入模块
    __import__(mod_str)

    # 利用getattr 函数得到 模块中的类
    return getattr(sys.modules[mod_str],class_str)


if __name__ == "__main__":
    # 测试
    import_str = "mmpc.Predictor.Predictor"
    DynamicClass = import_class(import_str)

    predictor = DynamicClass()
    
    columns={}
    with open("mmpc/columns.json", encoding='utf-8') as a:
        columns = json.load(a)
    
    #加载数据
    df = pd.read_csv('/PATH/TO/TEST_DATA') # 可以自行添加csv文件做测试
    syms = df['sym'].unique()
    dates = df['date'].unique()
    predict_result = []
    slide_window = 100
    for sym in syms:
        for date in dates:
            np_data = df[(df['sym'] == sym)&(df['date']==date)]
            np_data = np_data[columns['feature']].copy()
            for index in range(0, len(np_data) - slide_window):
                data = np_data.iloc[index : index + slide_window,:].copy()
                r = predictor.predict(data)
                print(r)
            break
        break
                    

    


