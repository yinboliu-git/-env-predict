import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import font_manager
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from ctrl_models import xlf_dict
from print_data import print_data

plt.style.use('seaborn-whitegrid')
my_font = font_manager.FontProperties(fname="/Library/Fonts/Songti.ttc")
import scipy.interpolate as spi
import math
import numpy
import matplotlib.pyplot as plt
from keras import regularizers
from keras.callbacks import ModelCheckpoint, Callback
from pandas import read_csv
from keras.models import Sequential, Model
from keras.layers import Dense, Concatenate, Conv2D, Conv1D, Bidirectional, Activation, Reshape, MaxPooling1D, LeakyReLU
from keras.layers import LSTM, Dropout, Input, Flatten
from keras.utils.vis_utils import plot_model
import joblib
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os


from keras import backend as K


def custom_activation(x):
    return K.sigmoid(x-1)

from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({'my_act': Activation(custom_activation)})


def use_FNN_func(x_train):
    input_shape = (x_train.shape[1],)
    x_in = Input(input_shape)
    x = Dense(units=512)(x_in)
    # x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    # x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x_out = Dense(2)(x)
    x_out = Activation('sigmoid')(x_out)
    # x_out = Activation('sigmoid')(x)
    model = Model(inputs=x_in, outputs=x_out)
    plot_model(model, to_file=sys._getframe().f_code.co_name + ".png", dpi=300, show_shapes=True)
    return model


def use_CNN_func(x_train):
    input_shape = (x_train.shape[1],)
    x_in = Input(input_shape)
    x = Dense(units=512)(x_in)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape(target_shape=(-1, 1))(x)
    x = Conv1D(256, 3, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Reshape(target_shape=(-1,))(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(2)(x)
    x_out = Activation('sigmoid')(x)
    model = Model(inputs=x_in, outputs=x_out)
    plot_model(model, to_file=sys._getframe().f_code.co_name + ".png", dpi=300, show_shapes=True)
    return model


def use_LSTM_func(x_train):
    input_shape = (x_train.shape[1],)
    x_in = Input(input_shape)
    x = Dense(units=512)(x_in)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape(target_shape=(-1, 1))(x)
    x = Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)), )(x)
    x = LeakyReLU(alpha=0.2)(x)
    # x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Reshape(target_shape=(-1,))(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(2)(x)
    x_out = Activation('sigmoid')(x)
    model = Model(inputs=x_in, outputs=x_out)
    plot_model(model, to_file=sys._getframe().f_code.co_name + ".png", dpi=300, show_shapes=True)
    return model


def get_data(x_data, y_data, jk_list=None):
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    if type(jk_list) is type(None):
        x_max_min = (x_data - x_data.min(axis=0)) / (x_data.max(axis=0) - x_data.min(axis=0))
        onehot_encoder = OneHotEncoder(sparse=False)
        y_data = y_data.reshape((-1, 1))
        y_onehot = onehot_encoder.fit_transform(y_data)
        return x_max_min, y_onehot

    x_max_min = (x_data - x_data.min(axis=0)) / (x_data.max(axis=0) - x_data.min(axis=0))
    jk_list = list(np.array(jk_list).reshape(-1))
    x_max_min = x_max_min[:, jk_list]
    onehot_encoder = OneHotEncoder(sparse=False)
    y_data = y_data.reshape((-1,1))
    y_onehot = onehot_encoder.fit_transform(y_data)

    return x_max_min, y_onehot


def mm_std(X):
    X = np.array(X)
    X = (X - np.mean(X,axis=0))/np.var(X,axis=0)
    X = (X - X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
    return X.tolist()


def use_model(x_data, y_data, model_dict={}, e=2, bs=16, save_file='./dnn_model_auc/'):
    # Seed value
    # Apparently you may use different seed values at each stage
    seed_value = 43

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

    y_data = np.array(y_data)
    x_data = np.array(x_data)

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    auc_dict = {}
    acc_dict = {}
    save_scores = {}
    history_loss = {}
    y_data_one = y_data[:,-1]
    unique, count = np.unique(y_data_one, return_counts=True)
    data_count = dict(zip(unique, count))
    all_cont = y_data_one.shape[0]
    print(f'正负样本比例：{data_count[1]/all_cont}:{data_count[0]/all_cont}')

    def myauc(y_true, y_pred):
        auc_vlue = tf.metrics.auc(y_true, y_pred,num_thresholds=498)[1]
        K.get_session().run(tf.local_variables_initializer())
        return auc_vlue

    class LossHistory(Callback):

        def __init__(self, x_test, y_test, all_epoch=1000):
            super().__init__()
            self.x_test = x_test
            self.y_test = y_test
            self.all_epoch = all_epoch

        def on_train_begin(self, logs={}):
            self.losses = []
            self.val_losses = []
            self.val_auc = []
            # self.val_myauc = []

        def on_epoch_end(self, epoch, logs=None):
            y_s = self.model.predict(self.x_test)
            fpr, tpr, thresholds = roc_curve(self.y_test[:,-1], y_s[:, -1])

            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.val_auc.append(auc(fpr, tpr))
            sys.stdout.write('\r完成epochs: {}/{}'
                             .format(epoch+1, self.all_epoch))  # \r 默认表示将输出的内容返回到第一个指针，这样的话，后面的内容会覆盖前面的内容。
            sys.stdout.flush()

            # self.val_myauc.append(logs.get('val_myauc'))
            # print(self.val_auc[-1], self.val_myauc[-1])

        def on_train_end(self, logs=None):
            print(' 完成...')

    for clf_name in model_dict.keys():
        print(clf_name + '正在运行')
        auc_dict[clf_name] = []
        acc_dict[clf_name] = []
        save_scores[clf_name+'score'] = []
        save_scores[clf_name+'true'] = []
        save_scores[clf_name + 'mm_std_score'] = []
        history_loss[clf_name] = {'train':[], 'val':[], 'val_auc':[]}
        num = 1
        for tr_idx, val_idx in kfold.split(x_data, y_data_one):
            print(f'算法{clf_name}的第{num}/{10}折交叉验证')
            num += 1
            train_x, train_y = x_data[tr_idx], y_data[tr_idx]
            test_x, test_y = x_data[val_idx], y_data[val_idx]

            clf = model_dict[clf_name](x_data)
            clf.compile(optimizer='adam', loss='mse', metrics=[myauc]) # categorical_crossentropy

            #  class_weight={0:1-data_count[0]/all_cont, 1:1-data_count[1]/all_cont}
            history = LossHistory(test_x,test_y)
            clf.fit(train_x, train_y,
                    validation_data=(test_x, test_y),epochs=e, batch_size=bs, verbose=0,
                    callbacks=[history])
            history_loss[clf_name]['train'].append(history.losses)   ## 获取loss
            history_loss[clf_name]['val'].append(history.val_losses)  ## 获取val_loss
            history_loss[clf_name]['val_auc'].append(history.val_auc)  ## 获取val_loss
            y_score = clf.predict(test_x)
            # y_score = clf.predict_proba(test_x)
            # idx_all = print_data(test_y, y_pred, y_score)
            y_s = y_score[:, -1]
            y_true = test_y[:, -1]
            y_p = y_s>0.5
            y_p = y_p.astype('int32')
            TN, FP, FN, TP = confusion_matrix(y_true, y_p).ravel()
            ACC = (TP + TN) / (TP + FP + FN + TN)
            fpr, tpr, thresholds = roc_curve(y_true, y_s)
            auroc = auc(fpr, tpr)

            save_scores[clf_name+'score'].extend(y_s.tolist())
            save_scores[clf_name+'true'].extend(y_true.tolist())
            save_scores[clf_name + 'mm_std_score'].extend(mm_std(y_s.tolist()))

            auc_dict[clf_name].append(auroc)
            acc_dict[clf_name].append(ACC)
            del clf
            # break
        print(auc_dict[clf_name])
        print(acc_dict[clf_name])
        auc_data = pd.DataFrame(auc_dict)
        print(auc_data.mean(axis=0))
    auc_data = pd.DataFrame(auc_dict)
    acc_data = pd.DataFrame(acc_dict)
    print(auc_data)
    print(acc_data)
    print()
    print('auc:')
    print(auc_data.mean(axis=0))
    print('acc:')
    print(acc_data.mean(axis=0))
    if not os.path.exists(save_file):
        os.mkdir(save_file)
    auc_data.to_csv(save_file + 'fold_pred_auc.csv', index=None)
    print()
    auc_mean = pd.DataFrame(auc_data.mean(axis=0))
    auc_mean.set_index(auc_data.keys(), inplace=True)
    auc_mean = auc_mean.T
    joblib.dump(history_loss, save_file + 'history_loss.dict')
    auc_mean.to_csv(save_file + 'mean_pred_auc.csv', index=None)

    save_scores = pd.DataFrame(save_scores)
    save_scores.to_csv(save_file + 'sores_ture.csv', index=None)
    auc_mean = auc_data.mean(axis=0)

    print(auc_mean)


def use_model_rf(x_data, y_data, model_dict={}, e=2, bs=16, save_file='./dnn_model_auc/'):

    y_data = np.array(y_data)
    x_data = np.array(x_data)

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    auc_dict = {}
    save_scores = {}
    history_loss = {}
    y_data_one = y_data[:,-1]
    unique, count = np.unique(y_data_one, return_counts=True)
    data_count = dict(zip(unique, count))
    all_cont = y_data_one.shape[0]
    print(f'正负样本比例：{data_count[1]/all_cont}:{data_count[0]/all_cont}')

    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))

    for clf_name in model_dict.keys():
        print(clf_name + '正在运行')
        auc_dict[clf_name] = []
        save_scores[clf_name+'score'] = []
        save_scores[clf_name+'true'] = []
        history_loss[clf_name] = {}
        num = 1
        for tr_idx, val_idx in kfold.split(x_data, y_data_one):
            history_loss[clf_name][num] = None
            train_x, train_y = x_data[tr_idx], y_data[tr_idx]
            test_x, test_y = x_data[val_idx], y_data[val_idx]

            clf = model_dict[clf_name](x_data)
            rf_clf = xlf_dict['rf']()
            clf.compile(optimizer='adam', loss='mse') # categorical_crossentropy

            #  class_weight={0:1-data_count[0]/all_cont, 1:1-data_count[1]/all_cont}
            history = LossHistory()
            clf.fit(train_x, train_y,
                    validation_data=(test_x, test_y),epochs=e, batch_size=bs, verbose=2,
                    callbacks=[history])

            rf_clf.fit(clf.predict(train_x), train_y[:,-1])
            history_loss[clf_name][num] = history.losses  ## 获取loss
            # y_score = clf.predict(test_x)
            y_score = rf_clf.predict_proba(clf.predict(test_x))
            # y_score = clf.predict_proba(test_x)
            # idx_all = print_data(test_y, y_pred, y_score)
            y_s = y_score[:, -1]
            y_true = test_y[:, -1]
            fpr, tpr, thresholds = roc_curve(y_true, y_s)
            auroc = auc(fpr, tpr)

            save_scores[clf_name+'score'].extend(y_s.tolist())
            save_scores[clf_name+'true'].extend(y_true.tolist())
            auc_dict[clf_name].append(auroc)
            del clf
        print(auc_dict[clf_name])
    auc_data = pd.DataFrame(auc_dict)

    print(auc_data)
    print()
    print(auc_data.mean(axis=0))
    if not os.path.exists(save_file):
        os.mkdir(save_file)
    auc_data.to_csv(save_file + 'fold_pred_auc.csv', index=None)
    print()
    auc_mean = pd.DataFrame(auc_data.mean(axis=0))
    auc_mean.set_index(auc_data.keys(), inplace=True)
    auc_mean = auc_mean.T
    joblib.dump(history_loss, save_file + 'history_loss.dict')
    auc_mean.to_csv(save_file + 'mean_pred_auc.csv', index=None)

    save_scores = pd.DataFrame(save_scores)
    save_scores.to_csv(save_file + 'sores_ture.csv', index=None)
    auc_mean = auc_data.mean(axis=0)

    print(auc_mean)

xlf_dict = {'fnn':use_FNN_func,
 'cnn':use_CNN_func}

def factor_train_dnn(x_data, y_data, jk_list, clf_name='fnn', e=100, bs=32):
    print('正在选择最优模型...')

    # data = pd.read_excel('./data/data_.xlsx')
    # y_data = np.array(data['y'])
    # x_data = np.array(data.iloc[:, 4:])
    save_file = './factor_or_no/dl_'
    y_data = np.array(y_data)
    x_data = np.array(x_data)
    x_max_min = (x_data - x_data.min(axis=0)) / (x_data.max(axis=0) - x_data.min(axis=0))
    jk_list = list(np.array(jk_list).reshape(-1))
    # x_max_min = x_max_min[:,jk_list]

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    auc_dict_factor = {}
    auc_dict_no_factor = {}
    for jk_i in jk_list:
        print(clf_name + '正在运行')
        auc_dict_factor[jk_i] = []
        auc_dict_no_factor[jk_i] = []
        for tr_idx, val_idx in kfold.split(x_max_min, y_data[:,-1]):
            train_x, train_y = x_max_min[tr_idx], y_data[tr_idx]
            test_x, test_y = x_max_min[val_idx], y_data[val_idx]

            clf1 = xlf_dict[clf_name](train_x[:, jk_i:jk_i + 1])
            clf1.compile(optimizer='adam', loss='mse')  # categorical_crossentropy

            clf1.fit(train_x[:, jk_i:jk_i + 1], train_y, epochs=e, batch_size=bs, verbose=0)
            y_score = clf1.predict(test_x[:, jk_i:jk_i + 1])
            # y_score = clf1.predict_proba(test_x[:, jk_i:jk_i + 1])
            y_pred = y_score[:,-1]>0.5
            fpr, tpr, thresholds = roc_curve(test_y[:,-1], y_score[:,-1])
            auroc = auc(fpr, tpr)
            auc_dict_factor[jk_i].append(auroc)

            jk_temp_list = [xx for xx in jk_list if xx != jk_i]
            clf2 = xlf_dict[clf_name](train_x[:, jk_temp_list])
            clf2.compile(optimizer='adam', loss='mse')  # categorical_crossentropy

            clf2.fit(train_x[:, jk_temp_list], train_y, epochs=e, batch_size=bs, verbose=0)
            y_score = clf2.predict(test_x[:, jk_temp_list])
            y_pred = y_score[:,-1]>0.5
            y_pred = y_pred.astype('int32')
            fpr, tpr, thresholds = roc_curve(test_y[:,-1], y_score[:,-1])
            auroc = auc(fpr, tpr)
            auc_dict_no_factor[jk_i].append(auroc)

        # print(auc_dict_factor[jk_i])
    auc_data_factor = pd.DataFrame(auc_dict_factor)
    auc_data_no_factor = pd.DataFrame(auc_dict_no_factor)

    print(auc_data_factor, auc_data_no_factor)
    print(auc_data_factor.mean(axis=0), auc_data_no_factor.mean(axis=0))
    if not os.path.exists(save_file):
        os.mkdir(save_file)
    auc_data_factor.to_csv(save_file + 'factor.csv', index=None)
    auc_mean = auc_data_factor.mean(axis=0)
    auc_mean.to_csv(save_file + 'factor_mean.csv', index=None)

    auc_data_no_factor.to_csv(save_file + 'no_factor.csv', index=None)
    auc_mean = auc_data_no_factor.mean(axis=0)
    auc_mean.to_csv(save_file + 'no_factor_mean.csv', index=None)
    return 0


if __name__ == '__main__':
    pass





