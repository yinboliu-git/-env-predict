from pipeline import use_best_model, factor_train, get_best_model, save_best_model
import numpy as np
import pandas as pd
import joblib
file_name = '2020s'

data = pd.read_excel('./data/data_.xlsx')
y_data = np.array(data['y'])

x_data = np.array(data.iloc[:, 4:])
x_data = np.array(x_data)
pca = joblib.load('./best_feature/pca.pkl')
x_max_min = np.array((x_data - x_data.min(axis=0)) / (x_data.max(axis=0) - x_data.min(axis=0)))
x_data = pca.transform(x_max_min)
print(f'pca降维至{x_data.shape[-1]}')
best_model = get_best_model(x_data, y_data, save_file ='./get_best_model_pca_auc/')
# save_best_model(x_data, y_data, 'rf', save_file_name='pca_')
print('结束...')
