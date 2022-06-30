from pipeline import use_best_model, factor_train, get_best_model
import numpy as np
import pandas as pd

file_name = '2020s'

jk = pd.read_csv('./best_feature/best_features.csv', header=None)

data = pd.read_excel('./data/data_.xlsx')
y_data = np.array(data['y'])
x_data = np.array(data.iloc[:, 4:])

best_model = get_best_model(x_data, y_data, jk, save_file ='./get_best_model_auc/')
print('best_model:' + best_model)
print('结束...')
