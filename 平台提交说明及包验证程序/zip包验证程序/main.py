import sys
import pandas as pd
import json
import torch

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
    df = pd.read_csv('D:\my-dev-code\FBDQA-final-project\FBDQA2021A_MMP_Challenge_ver0.2\data\snapshot_sym9_date7_am.csv') # 可以自行添加csv文件做测试
    syms = df['sym'].unique()
    dates = df['date'].unique()
    predict_result = []
    slide_window = 100
    for sym in syms:
        for date in dates:
            correct_test_preds = 0
            total_test_preds = 0

            target0_num = 0
            target1_num = 0
            target2_num = 0
            target0_num_predict = 0
            target1_num_predict = 0
            target2_num_predict = 0
            np_data = df[(df['sym'] == sym)&(df['date']==date)]
            # np_data = np_data[columns['feature']].copy()
            for index in range(0, len(np_data) - slide_window):
                data = np_data.iloc[index : index + slide_window,:].copy()
                target = data.iloc[-1]["label_5"]
                if target == 0:
                    target0_num += 1
                elif target == 1:
                    target1_num += 1
                else:
                    target2_num += 1
                outputs = predictor.predict(data)   # [label_num]
                
                if outputs[0] == 0:
                    target0_num_predict += 1
                elif outputs[0] == 1:
                    target1_num_predict += 1
                else:
                    target2_num_predict += 1
                if outputs[0] == target:
                    correct_test_preds += 1
                total_test_preds += 1
            print(f"sym: {sym}, date: {date}, acc: {correct_test_preds / total_test_preds}")
            print(f"correct_test_preds: {correct_test_preds}, total_test_preds: {total_test_preds}")
            # calculate target percentage
            print(f"target0_num: {target0_num}, target1_num: {target1_num}, target2_num: {target2_num}")
            print(f"target0_percentage: {target0_num / total_test_preds}, target1_percentage: {target1_num / total_test_preds}, target2_percentage: {target2_num / total_test_preds}")
            print(f"target0_num_predict: {target0_num_predict}, target1_num_predict: {target1_num_predict}, target2_num_predict: {target2_num_predict}")
            print(f"target0_predict_percentage: {target0_num_predict / total_test_preds}, target1_predict_percentage: {target1_num_predict / total_test_preds}, target2_predict_percentage: {target2_num_predict / total_test_preds}")
            # target 是0或2里面判断正确的比例
            print(f"target0_acc: {target0_num_predict / target0_num}, target2_acc: {target2_num_predict / target2_num}")
            break
        break
                    

    


