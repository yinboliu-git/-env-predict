from 深度学习 import 补充实验使用深度学习 as cnn
import numpy as np
import pandas as pd
import joblib
from pipeline import save_best_model
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model
import os

def train_nn_auc(epochs=100):
    data = pd.read_excel('./data/data_.xlsx')
    y_data = np.array(data['y'])

    x_data = np.array(data.iloc[:, 4:])
    x_data = np.array(x_data)
    pca = joblib.load('./best_feature/pca.pkl')
    x_max_min = np.array((x_data - x_data.min()) / (x_data.max() - x_data.min()))
    x_data = pca.transform(x_max_min)
    x_data, y_data = cnn.get_data(x_data, y_data)

    model1 = cnn.use_FNN_func
    model2 = cnn.use_CNN_func
    model3 = cnn.use_LSTM_func
    # model_dict = {'fnn':model1, 'cnn':model2,'lstm':model3}
    model_dict = {'fnn':model1,'cnn':model2}
    cnn.use_model(x_data, y_data, model_dict,e=epochs,bs=32, save_file='./dnn_model_pca_auc/')

    # pca = joblib.load('./best_feature/pca.pkl')
    # x_mm = pca.transform(x_max_min)
    # save_best_model(x_mm, y_data, best_model_name='')
    # get_pca_features(x_data, methods=i)
    return 0

def save_nn(modelname,epochs=100):
    data = pd.read_excel('./data/data_.xlsx')
    y_data = np.array(data['y'])

    x_data = np.array(data.iloc[:, 4:])
    x_data = np.array(x_data)
    pca = joblib.load('./best_feature/pca.pkl')
    x_data = np.array((x_data - x_data.min(axis=0)) / (x_data.max(axis=0) - x_data.min(axis=0)))
    x_data = pca.transform(x_data)
    x_data, y_data = cnn.get_data(x_data, y_data)

    model1 = cnn.use_FNN_func(x_data)
    model2 = cnn.use_CNN_func(x_data)
    model3 = cnn.use_LSTM_func(x_data)
    model_dict = {'fnn':model1, 'cnn':model2, 'lstm':model3}
    # model_dict = {'fnn':model1}
    # cnn.use_model(x_data, y_data, model_dict,e=10,bs=32)

    unique, count = np.unique(y_data[:,-1], return_counts=True)
    data_count = dict(zip(unique, count))
    all_cont = y_data[:,-1].shape[0]
    # print(f'正负样本比例：{data_count[1]/all_cont}:{data_count[0]/all_cont}')

    clf = model_dict[modelname]
    clf.compile(optimizer='adam', loss='mse') # categorical_crossentropy

    clf.fit(x_data, y_data, epochs=epochs, batch_size=32, verbose=2)
    a = clf.predict(x_data)
    print(a)
    clf.save('models/model_best_pca_'+modelname+'.pkl')


    # pca = joblib.load('./best_feature/pca.pkl')
    # x_mm = pca.transform(x_max_min)
    # save_best_model(x_mm, y_data, best_model_name='')
    # get_pca_features(x_data, methods=i)
    return 0

def use_nn(file_name, best_model):
    save_name = file_name
    mname = best_model

    data = pd.read_excel('./data/data_.xlsx')
    # './pred_data/' + file_name + '.xlsx'
    pred_data = pd.read_excel('./pred_data/' + file_name + '.xlsx')

    y_data = np.array(data['y'])
    x_data = np.array(data.iloc[:, 4:])
    x_pred_data = np.array(pred_data.iloc[:, 3:])
    x_sits = pred_data.iloc[:, 1:3]

    x_pred_data = (x_pred_data - x_data.min(axis=0)) / (x_data.max(axis=0) - x_data.min(axis=0))
    x_data = np.array((x_data - x_data.min(axis=0)) / (x_data.max(axis=0) - x_data.min(axis=0)))


    pca = joblib.load('./best_feature/pca.pkl')
    x_data = pca.transform(x_data)
    x_pred_data = pca.transform(x_pred_data)
    x_pred_data = (x_pred_data-x_data.min(axis=0))/(x_data.max(axis=0) - x_data.min(axis=0))
    clf = load_model('models/model_best_pca_'+best_model+'.pkl')
    y_p = clf.predict(x_pred_data)
    y_proba = y_p
    y_p = y_p[:,-1]/np.sum(y_p, axis=1) > 0.5
    y_p = y_p.astype('int32')

    y_p = pd.DataFrame(y_p, columns=['等级'])
    y_proba = pd.DataFrame(y_proba)
    y_proba_cls = np.array(y_proba.iloc[:, -1]/np.sum(y_proba,axis=1))
    # y_proba_cls = np.array(y_proba.iloc[:, 1])
    # y_proba_cls = (y_proba_cls - y_proba_cls.min())/(y_proba_cls.max() - y_proba_cls.min())
    # y_proba_cls[y_proba_cls > 0.75] = 3
    # y_proba_cls[(0.75 >= y_proba_cls) & (y_proba_cls >= 0.5)] = 2
    # y_proba_cls[(0.5 >= y_proba_cls) & (y_proba_cls >= 0.25)] = 1
    # y_proba_cls[0.25 >= y_proba_cls] = 0

    y_proba_cls[y_proba_cls > 0.75] = 3
    y_proba_cls[(0.75 >= y_proba_cls) & (y_proba_cls >= 0.5)] = 2
    y_proba_cls[(0.5 >= y_proba_cls) & (y_proba_cls >= 0.25)] = 1
    y_proba_cls[0.25 >= y_proba_cls] = 0


    y_proba_cls = pd.DataFrame(y_proba_cls, columns=['等级'])
    y_proba_score = pd.DataFrame(np.array(y_proba.iloc[:, -1]), columns=['得分'])
    if not os.path.exists('./results'):
        os.mkdir('./results')
    if not os.path.exists('./results/pca_dnn'):
        os.mkdir('./results/pca_dnn')
    save_file = './results/pca_dnn/'+mname
    print(y_p)
    print(y_proba)
    print('预测完成..')
    x_sits = pd.DataFrame(x_sits)
    x_sits.columns = ['X', 'Y']
    y_proba_cls = pd.concat((x_sits, y_proba_cls), axis=1)
    y_proba_cls.to_csv(save_file+'-sites_4class' + save_name + '.csv', index=None)
    y_p = pd.concat((x_sits, y_p), axis=1)
    y_p.to_csv(save_file +'-sites_2cls' + save_name + '.csv', index=None)
    y_proba_score = pd.concat((x_sits, y_proba_score), axis=1)
    y_proba_score.to_csv(save_file +'-sites_scores' + save_name + '.csv', index=None)

    ## 保存在data
    # y_proba_cls.to_csv('c:/data/csv_data/'+mname+'-sites_class' + save_name + '.csv', index=None)
    # y_p.to_csv('c:/data/csv_data/'+ mname +'-sites_cls' + save_name + '.csv', index=None)

    print(file_name+'结束...')
    print(file_name+'结束...')

    return y_proba_cls

if __name__ == '__main__':
    train_nn_auc(epochs=100)
    # save_nn('fnn', epochs=100)
    # file_list = ['2020s', '2030s', '2050s', '2070s']
    # for f in file_list:
    #     use_nn(file_name=f, best_model='fnn')
