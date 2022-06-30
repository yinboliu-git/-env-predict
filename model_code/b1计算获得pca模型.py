from pipeline import get_pca_features
import numpy as np
import pandas as pd


def predict_pred_data():
    data = pd.read_excel('./data/data_.xlsx')
    y_data = np.array(data['y'])
    x_data = np.array(data.iloc[:, 4:])

    i = get_pca_features(x_data, y_data)
    get_pca_features(x_data, methods=i)  # 15

    return 0


if __name__ == '__main__':
    predict_pred_data()
