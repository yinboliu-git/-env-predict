from pipeline import use_best_model, factor_train
import numpy as np
import pandas as pd


def predict_pred_data(file_name):
    data = pd.read_excel('./data/data_.xlsx')
    pred_data = pd.read_excel('./pred_data/'+file_name+'.xlsx')
    best_feature = pd.read_csv('./best_feature/best_features.csv',header=None)

    y_data = np.array(data['y'])
    x_data = np.array(data.iloc[:, 4:])
    x_pred_data = np.array(pred_data.iloc[:, 3:])
    sites = pred_data.iloc[:, 1:3]

    best_model_path = './models/gbrt.pkl'
    y_proba_cls = use_best_model(sites, x_pred_data, x_data, best_model_path, best_feature,save_name=file_name)
    print(file_name+'结束...')

    return y_proba_cls


if __name__ == '__main__':

    file_list = ['2020s', '2030s', '2050s', '2070s']
    for f in file_list:
        predict_pred_data(f)
