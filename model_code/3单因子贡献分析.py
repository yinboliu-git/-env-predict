from pipeline import one_to_one_factor,one_factor,factor_train
import numpy as np
import pandas as pd


def predict_pred_data(file_name):
    data = pd.read_excel('./data/data_.xlsx')
    # pred_data = pd.read_excel('./pred_data/'+file_name+'.xlsx')
    best_feature = pd.read_csv('./best_feature/best_features.csv')

    y_data = np.array(data['y'])
    x_data = np.array(data.iloc[:, 4:])

    best_model_path = './models/gbrt.pkl'
    # y_proba_cls = use_best_model(sites, x_pred_data, x_data, best_model_path, best_feature,save_name=file_name)
    factor_train(x_data, y_data, best_feature, clf_name=file_name)
    # one_to_one_factor(x_data,y_data,best_feature,model_name='rf',save_name=file_name,number_density=1000)
    print(file_name+'结束...')


if __name__ == '__main__':
    file_name = 'gbrt'
    predict_pred_data(file_name)
