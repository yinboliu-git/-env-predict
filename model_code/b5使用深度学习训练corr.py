from ctrl_models import xlf_dict
from 深度学习 import 补充实验使用深度学习 as cnn
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model
import os

from keras import backend as K

seed_value = 42

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np

np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf

tf.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def custom_activation(x):
    return K.tanh(x+0.5)/2 + 0.5



def train_nn_auc(epochs=100):
    data = pd.read_excel('./data/data_.xlsx')
    y_data = np.array(data['y'])
    jk = pd.read_csv('./best_feature/best_features.csv')
    x_data = np.array(data.iloc[:, 4:])
    x_data = np.array(x_data)
    x_data, y_data = cnn.get_data(x_data, y_data,jk)

    model1 = cnn.use_FNN_func
    model2 = cnn.use_CNN_func
    model3 = cnn.use_LSTM_func
    # model_dict = {'fnn':model1, 'cnn':model2,'lstm':model3}
    model_dict = {'fnn':model1, 'cnn':model2}
    cnn.use_model(x_data, y_data, model_dict,e=epochs,bs=32)

    # pca = joblib.load('./best_feature/pca.pkl')
    # x_mm = pca.transform(x_max_min)
    # save_best_model(x_mm, y_data, best_model_name='')
    # get_pca_features(x_data, methods=i)
    return 0

def save_nn(model='fnn', epochs=100):
    data = pd.read_excel('./data/data_.xlsx')
    y_data = np.array(data['y'])
    jk = pd.read_csv('./best_feature/best_features.csv')
    x_data = np.array(data.iloc[:, 4:])
    x_data = np.array(x_data)
    x_data, y_data = cnn.get_data(x_data, y_data, jk)

    model1 = cnn.use_FNN_func(x_data)
    model2 = cnn.use_CNN_func(x_data)
    model3 = cnn.use_LSTM_func(x_data)
    model_dict = {'fnn':model1, 'cnn':model2, 'lstm':model3}

    unique, count = np.unique(y_data[:,-1], return_counts=True)
    data_count = dict(zip(unique, count))
    all_cont = y_data[:,-1].shape[0]
    # print(f'正负样本比例：{data_count[1]/all_cont}:{data_count[0]/all_cont}')

    clf = model_dict[model]

    clf.compile(optimizer='adam', loss='mse') # categorical_crossentropy

    clf.fit(x_data, y_data, epochs=epochs, batch_size=32, verbose=0)
    aa = np.array(clf.predict(x_data))
    aa = aa[:,-1] > aa[:,0]
    aa = aa.astype('int32')
    print(aa)

    clf.save('models/smodel_best_'+model+'.pkl')

    # pca = joblib.load('./best_feature/pca.pkl')
    # x_mm = pca.transform(x_max_min)
    # save_best_model(x_mm, y_data, best_model_name='')
    # get_pca_features(x_data, methods=i)
    return 0

def use_nn(file_name, best_model):
    save_name = file_name
    mname = best_model

    data = pd.read_excel('./data/data_.xlsx')
    pred_data = pd.read_excel('./pred_data/'+file_name+'.xlsx')
    jk_list = pd.read_csv('./best_feature/best_features.csv')
    jk_list = list(np.array(jk_list).reshape(-1))

    y_data = np.array(data['y'])
    x_data = np.array(data.iloc[:, 4:])
    x_pred_data = np.array(pred_data.iloc[:, 3:])
    # x_pred_data = x_data
    x_sits = pred_data.iloc[:, 1:3]

    x_pred_data = (x_pred_data - x_data.min(axis=0)) / (x_data.max(axis=0) - x_data.min(axis=0))
    x_pred_data = x_pred_data[:,jk_list]

    clf = load_model('models/smodel_best_'+best_model+'.pkl')
    y_p = clf.predict(x_pred_data)
    y_proba = y_p
    y_p = y_p[:,-1] > 0.5
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
    if not os.path.exists('./results/corr_dnn'):
        os.mkdir('./results/corr_dnn')
    save_file = './results/corr_dnn/'+mname
    # y_p.to_csv('./results/corr_dnn/'+mname+'-y_pred' + save_name + '.csv')
    # y_proba.to_csv('./results/corr_dnn/'+mname+'-y_scores' + save_name + '.csv')
    # y_proba_cls.to_csv('./results/corr_dnn/'+mname+'-y_class' + save_name + '.csv', index=None)
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

    return y_proba_cls

if __name__ == '__main__':
    # train_nn_auc(epochs=100)
    save_nn('cnn',epochs=1)
    # file_list = ['2020s', '2030s', '2050s', '2070s']
    # for f in file_list:
    #     use_nn(file_name=f, best_model='fnn')
