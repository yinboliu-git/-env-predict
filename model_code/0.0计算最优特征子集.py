from pipeline import get_jickknife, get_corr, get_best_features
import numpy as np
import pandas as pd

data = pd.read_excel('./data/data_.xlsx')
y_data = np.array(data['y'])
x_data = np.array(data.iloc[:, 4:])

x_corr = get_corr(x_data)
# jk_value = get_jickknife(x_data, y_data)
# best_feature = get_best_features(x_corr, jk_value)
# print(best_feature)