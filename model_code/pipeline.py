import sys

import joblib
import seaborn as sns
import numpy as np
from sklearn.model_selection import ParameterGrid, StratifiedKFold, LeaveOneOut
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, auc
import pandas as pd
from print_data import print_data
import os
from ctrl_models import xlf_dict


def get_corr(x_data, corr_csv_filename='./corr/x_corr.csv', corr_img_filename='./corr/corr.png'):
    print('正在进行相关系数计算....')
    # data = pd.read_excel('./data/data_.xlsx')
    # y_data = np.array(data['y'])
    # x_data = np.array(data.iloc[: ,4:])
    if not os.path.exists('./corr/'):
        os.mkdir('./corr')

    x_data = np.array(x_data)
    x_max_min = (x_data - x_data.min(axis=0)) / (x_data.max(axis=0) - x_data.min(axis=0))
    x_max_min = pd.DataFrame(x_max_min)
    x_corr = x_max_min.corr().abs()

    font0 = {
        'weight': 'medium',
        'size': 21,
        "fontweight": 'bold',
    }
    font1 = {
        'weight': 'medium',
        'size': 13,
        "fontweight": 'bold',
    }

    x_corr_df = pd.DataFrame(np.array(x_corr), columns=[x for x in range(1, x_corr.shape[1] + 1)],
                             index=[x for x in range(1, x_corr.shape[1] + 1)])
    x_corr_df.to_csv(corr_csv_filename)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=600)
    # 二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
    # 和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
    cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
    xl = ['BIO' + str(i+1) for i in range(x_corr_df.shape[0])]
    yl = xl
    # sns.heatmap(df, linewidths = 0.05, vmax=1, vmin=0)
    sns.heatmap(x_corr_df, annot=False, vmax=1, vmin=0, xticklabels=xl, yticklabels=yl, cmap="YlGnBu")

    # sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True,
    #            square=True, cmap="YlGnBu")
    # ax.set_title('二维数组热力图', fontsize = 18)

    # plt.xlabel(xlabel=, rotation=90)
    # plt.yticks(rotation=90)
    ax.set_xticklabels(xl, rotation=90)
    ax.set_yticklabels(yl, rotation=0)
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(17) for label in labels]
    [label.set_fontweight('bold') for label in labels]
    [label.set_fontstyle('normal') for label in labels]
    # ax.set_ylabel('Models', fontdict=font0)
    # ax.set_xlabel('Data', fontdict=font0) #横变成y轴，跟矩阵原始的布局情况是一样的
    plt.savefig(corr_img_filename, bbox_inches='tight')
    # x_corr.to_csv('./x_corr_.csv')
    return x_corr_df


def get_jickknife(x_data, y_data, jk_csv_filename='./jickknife_values/jk_value.csv', clf=xlf_dict['rf']):
    print('正在进行jickknife....')
    if not os.path.exists('./jickknife_values'):
        os.mkdir('jickknife_values')
    x_max_min = (x_data - x_data.min(axis=0)) / (x_data.max(axis=0) - x_data.min(axis=0))
    pred_dict = {}
    y_test_dict = {}
    auprc_dict = {}
    x_data = pd.DataFrame(x_data)
    header_range = x_data.keys().shape[0]
    loo_numbers = x_max_min.shape[0]
    for header_name in range(header_range):
        # print(str(header_name) + '正在运行')
        pred_dict[header_name] = []
        y_test_dict[header_name] = []

        loo = LeaveOneOut()
        i = 0
        for train_index, test_index in loo.split(x_max_min, y_data):
            train_x, test_x = x_max_min[train_index, header_name:header_name + 1], x_max_min[test_index,
                                                                                   header_name:header_name + 1]
            train_y, test_y = y_data[train_index], y_data[test_index]

            clf.fit(train_x, train_y)
            y_pred = (clf.predict_proba(test_x))[:, 1]
            # y_score = clf.predict_proba(test_x)
            pred_dict[header_name].append(y_pred.item())
            y_test_dict[header_name].append(test_y.item())
            i += 1
            # print('进行特征{}/{}, 此特征完成loo {}/{}'.format(header_name+1, header_range, i, loo_numbers))
            sys.stdout.write('\r进行特征: {}/{}, 此特征完成loo: {}/{}'.format(header_name + 1, header_range, i,
                                                                     loo_numbers))  # \r 默认表示将输出的内容返回到第一个指针，这样的话，后面的内容会覆盖前面的内容。
            sys.stdout.flush()
            # time.sleep(0.1)
        print()  ## 这里刷新后换行

        precision, recall, thresholds = precision_recall_curve(np.array(y_test_dict[header_name]),
                                                               np.array(pred_dict[header_name]))

        auprc = auc(recall, precision)
        auprc_dict[header_name + 1] = [auprc]

    print(auprc_dict)
    auprc_df = pd.DataFrame(auprc_dict)
    auprc_df.to_csv(jk_csv_filename, index=None)
    return auprc_df


def get_best_features(corr_data, jk_scores, best_features_filename='./best_feature/best_features.csv'):  # use
    print('正在获取最优特征...')
    if not os.path.exists('./best_feature'):
        os.mkdir('./best_feature')

    corr_data = pd.DataFrame(corr_data)
    idx_scores = np.array(jk_scores)
    idx_scores = idx_scores.reshape(-1)
    if len(idx_scores) != corr_data.keys().__len__():
        raise Exception('jk_scores 与 corr_data 的长度不匹配...')

    corr_data = corr_data > 0.8

    corr_list = {}
    arr_list = []
    for i in range(corr_data.shape[0]):
        if i not in arr_list:
            corr_list[i] = [i]
            for j in range(i + 1, corr_data.shape[1]):
                if corr_data.iloc[i, j] == True:
                    corr_list[i].append(j)
                    arr_list.append(j)

    best_list = []
    for i in corr_list.keys():
        idx_arg = np.argsort(idx_scores[corr_list[i]])
        best_list.append(corr_list[i][idx_arg[-1]])

    best_list_features_numbers = np.sort(best_list)
    save_data = pd.DataFrame(best_list_features_numbers)
    save_data = save_data.T
    save_data.to_csv(best_features_filename, index=None)
    print('最优特征组合jk是：')
    print(best_list_features_numbers)
    return best_list_features_numbers


import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys


def get_pca_features(x_data, y_data=None, methods='jack', clf=xlf_dict['rf'],
                     best_features_filename='./best_feature/pac_values.csv'):  # use
    print('正在获取最优特征...')
    if not os.path.exists('./best_feature'):
        os.mkdir('./best_feature')
    x_data = np.array(x_data)
    x_max_min = np.array((x_data - x_data.min(axis=0)) / (x_data.max(axis=0) - x_data.min(axis=0)))

    if methods == 'jack':
        y_data = np.array(y_data)
        pred_dict = {}
        y_test_dict = {}
        auprc_dict = {}
        x_data = pd.DataFrame(x_data)
        header_range = x_data.keys().shape[0]
        loo_numbers = x_max_min.shape[0]
        auprc_dict['max'] = 0
        auprc_dict['maxid'] = 0
        ct = x_data.shape[-1] + 1
        for i in range(1, ct):
            pca = PCA(i)
            pca.fit(x_max_min)
            lowDmat = pca.transform(x_max_min)  # 降维后的数据
            # score = pca.score_samples(x_data)

            pred_dict[i] = []
            y_test_dict[i] = []
            loo = LeaveOneOut()
            j = 0
            for train_index, test_index in loo.split(lowDmat, y_data):
                train_x, test_x = lowDmat[train_index], lowDmat[test_index]
                train_y, test_y = y_data[train_index], y_data[test_index]

                clf.fit(train_x, train_y)
                y_pred = (clf.predict_proba(test_x))[:, 1]
                # y_score = clf.predict_proba(test_x)
                pred_dict[i].append(y_pred.item())
                y_test_dict[i].append(test_y.item())
                # print('进行特征{}/{}, 此特征完成loo {}/{}'.format(header_name+1, header_range, i, loo_numbers))
                j += 1
                sys.stdout.write('\r进行pca: {}/{}, 此特征完成loo: {}/{}'.format(i, header_range, j,
                                                                          loo_numbers))  # \r 默认表示将输出的内容返回到第一个指针，这样的话，后面的内容会覆盖前面的内容。
                sys.stdout.flush()
                # time.sleep(0.1)
            print()  ## 这里刷新后换行ii

            precision, recall, thresholds = precision_recall_curve(np.array(y_test_dict[i]),
                                                                   np.array(pred_dict[i]))

            auprc = auc(recall, precision)
            auprc_dict[i] = [auprc]
            auprc_dict['maxid'] = [i] if auprc > auprc_dict['max'] else auprc_dict['maxid'] # 先
            auprc_dict['max'] = [auprc] if auprc > auprc_dict['max'] else auprc_dict['max'] # 后


        print(auprc_dict)
        auprc_df = pd.DataFrame(auprc_dict)
        auprc_df.to_csv(best_features_filename, index=None)
        return auprc_dict['maxid'][0]

    else:
        pca = PCA(methods)
        pca.fit(x_max_min)
        lowDmat = pca.transform(x_max_min)  # 降维后的数据
        joblib.dump(pca, './best_feature/pca.pkl')
        return lowDmat


def mm_std(X):
    X = np.array(X)
    X = (X - np.mean(X,axis=0))/np.var(X,axis=0)
    X = (X - X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
    return X.tolist()


def get_best_model(x_data, y_data, jk_list=None, save_file='./get_best_model_auc/'):
    print('正在选择最优模型...')

    # data = pd.read_excel('./data/data_.xlsx')
    # y_data = np.array(data['y'])
    # x_data = np.array(data.iloc[:, 4:])

    y_data = np.array(y_data)
    x_data = np.array(x_data)
    x_max_min = (x_data - x_data.min(axis=0)) / (x_data.max(axis=0) - x_data.min(axis=0))
    if not (type(jk_list) is type(None)):
        jk_list = list(np.array(jk_list).reshape(-1))
        x_max_min = x_max_min[:, jk_list]

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    auc_dict = {}
    acc_dict = {}
    save_scores = {}

    for clf_name in xlf_dict.keys():
        print(clf_name + '正在运行')
        auc_dict[clf_name] = []
        acc_dict[clf_name] = []
        save_scores[clf_name + 'score'] = []
        save_scores[clf_name + 'true'] = []
        save_scores[clf_name + 'mm_std_true'] = []
        for tr_idx, val_idx in kfold.split(x_max_min, y_data):
            train_x, train_y = x_max_min[tr_idx], y_data[tr_idx]
            test_x, test_y = x_max_min[val_idx], y_data[val_idx]
            try:
                clf = xlf_dict[clf_name]()
            except:
                clf = xlf_dict[clf_name]
            clf.fit(train_x, train_y)
            y_pred = clf.predict(test_x)
            y_score = clf.predict_proba(test_x)
            # idx_all = print_data(test_y, y_pred, y_score)
            y_p = y_pred
            TN, FP, FN, TP = confusion_matrix(test_y, y_p).ravel()
            ACC = (TP + TN) / (TP + FP + FN + TN)
            fpr, tpr, thresholds = roc_curve(test_y , y_score[:, 1])
            auroc = auc(fpr, tpr)

            save_scores[clf_name + 'score'].extend(y_score[:, -1].tolist())
            save_scores[clf_name + 'true'].extend(test_y.tolist())
            save_scores[clf_name + 'mm_std_true'].extend(mm_std(y_score[:, -1].tolist()))
            auc_dict[clf_name].append(auroc)
            acc_dict[clf_name].append(ACC)
        print(auc_dict[clf_name])
        print(acc_dict[clf_name])
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
    auc_mean = pd.DataFrame(auc_data.mean(axis=0))
    auc_mean.set_index(auc_data.keys(), inplace=True)
    auc_mean = auc_mean.T

    auc_mean.to_csv(save_file + 'mean_pred_auc.csv', index=None)

    save_scores = pd.DataFrame(save_scores)
    save_scores.to_csv(save_file + 'sores_ture.csv', index=None)
    auc_mean = auc_data.mean(axis=0)
    best_model_name = auc_mean[auc_mean == auc_mean.max()].keys().item()
    print('最优算法是：' + best_model_name)

    if not (type(jk_list) is type(None)):
        save_best_model(x_data, y_data, best_model_name, jk_list)
    else:
        save_best_model(x_data, y_data, best_model_name)
    return best_model_name


def save_best_model(x_data, y_data, best_model_name, jk_list=None, save_file_name='', best_param=''):
    print('正在保存最优模型...')
    # data = pd.read_excel('./data/data_.xlsx')
    # y_data = np.array(data['y'])
    # x_data = np.array(data.iloc[:, 4:])
    y_data = np.array(y_data)
    x_data = np.array(x_data)
    x_max_min = (x_data - x_data.min(axis=0)) / (x_data.max(axis=0) - x_data.min(axis=0))
    try:
        clf = xlf_dict[best_model_name]()
    except:
        clf = xlf_dict[best_model_name]
    if best_param != '':
        for i in best_param.keys():
            if not (hasattr(clf, i)):
                raise Exception('xlf_append: {} 属性在{}中不存在..'.format(i, clf))
            setattr(clf, i, best_param[i])
            ##最优参数示例： {'a':1,'b':2}
    if type(jk_list) != type(None):
        jk_list = list(np.array(jk_list).reshape(-1))
        x_max_min = x_max_min[:, jk_list]
    clf.fit(x_max_min, y_data)
    print(clf.predict(x_max_min))  # 自测准确度
    if not os.path.exists('./models/'):
        os.mkdir('./models/')

    joblib.dump(clf, './models/' + save_file_name + best_model_name + '.pkl')
    print('保存完成，请在models文件夹里查看..')


def use_best_model(x_sits, x_pred_data, x_data, model_path, jk_list, save_name='',best_model_name='gbrt'):
    print('正在预测新数据...')

    # data = pd.read_excel('./data/data_.xlsx')
    # pred_data = pd.read_excel('./pred_data/2030l.xlsx')
    # y_data = np.array(data['y'])
    # x_data = np.array(data.iloc[:, 4:])
    # x_pred_data = np.array(pred_data.iloc[:, 3:])
    x_pred_data = np.array(x_pred_data)
    x_data = np.array(x_data)
    x_max_min = (x_pred_data - x_data.min(axis=0)) / (x_data.max(axis=0) - x_data.min(axis=0))
    jk_list = list(np.array(jk_list).reshape(-1))
    x_max_min = x_max_min[:, jk_list]

    clf = joblib.load(model_path)
    y_p = clf.predict(x_max_min)
    y_proba = clf.predict_proba(x_max_min)
    y_p = pd.DataFrame(y_p, columns=['等级'])
    y_proba = pd.DataFrame(y_proba)
    y_proba_cls = np.array(y_proba.iloc[:, 1])

    y_proba_cls[y_proba_cls > 0.75] = 3
    y_proba_cls[(0.75 >= y_proba_cls) & (y_proba_cls >= 0.5)] = 2
    y_proba_cls[(0.5 >= y_proba_cls) & (y_proba_cls >= 0.25)] = 1
    y_proba_cls[0.25 >= y_proba_cls] = 0

    y_proba_cls = pd.DataFrame(y_proba_cls, columns=['等级'])
    y_proba_score = pd.DataFrame(np.array(y_proba.iloc[:, 1]), columns=['得分'])
    if not os.path.exists('./results'):
        os.mkdir('./results')
    if not os.path.exists('./results/corr_ml'):
        os.mkdir('./results/corr_ml')
    save_file = './results/corr_ml/' + best_model_name
    # y_p.to_csv('./results/y_pred' + save_name + '.csv')
    # y_proba.to_csv('./results/y_scores' + save_name + '.csv')
    # y_proba_cls.to_csv('./results/y_class' + save_name + '.csv', index=None)
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

    return y_proba_cls


def one_factor(x_data, y_data, jk_list, model_path, save_name='', number_density=100):
    print('单因子生存概率正在执行...')

    # data = pd.read_excel('./data/data_.xlsx')
    # pred_data = pd.read_excel('./pred_data/2030l.xlsx')
    # y_data = np.array(data['y'])
    # x_data = np.array(data.iloc[:, 4:])
    # x_pred_data = np.array(pred_data.iloc[:, 3:])

    y_data = np.array(y_data)
    x_data = np.array(x_data)

    x_max_min = (x_data - x_data.min(axis=0)) / (x_data.max(axis=0) - x_data.min(axis=0))
    x_max_min = x_max_min[y_data == 1, :]
    jk_list = list(np.array(jk_list).reshape(-1))
    x_max_min = x_max_min[:, jk_list]

    clf = joblib.load(model_path)
    y_proba = clf.predict_proba(x_max_min)
    max_exist_number = np.where(y_proba == ((y_proba[:, 1] - 0.5).min() + 0.5))[0].item()
    max_exist = y_proba[max_exist_number, 1]
    print('当前最大存在概率:{}'.format(max_exist))
    exist_row = x_max_min[max_exist_number, :]
    exist_row = exist_row.reshape(1, -1)
    exist_row = np.tile(exist_row, (number_density + 1, 1))
    exist_row = np.expand_dims(exist_row, axis=0)
    # exist_row = np.expand_dims(exist_row, axis=0)
    exist_row = np.tile(exist_row, (len(jk_list), 1, 1))

    pred_exist_one = {}
    pred_scope = {}
    x_data_max = x_data.max(axis=0)[jk_list]
    x_data_min = x_data.min(axis=0)[jk_list]
    for i in range(len(jk_list)):
        exist_temp = exist_row[i]
        exist_temp[:, i] = np.arange(0, 1 + 1 / number_density, 1 / number_density)
        pred_exist_one[jk_list[i]] = (clf.predict_proba(exist_temp))[:, 1]
        pred_scope[jk_list[i]] = (x_data_max[i] - x_data_min[i]) * exist_temp[:, i] + x_data_min[i]
    pred_exist_one_pd = pd.DataFrame(pred_exist_one)
    pred_scope_pd = pd.DataFrame(pred_scope)
    if not os.path.exists('./exist_probability'):
        os.mkdir('./exist_probability')
    pred_exist_one_pd.to_csv('./exist_probability/pred_exist_one' + save_name + '.csv', index=None)
    pred_scope_pd.to_csv('./exist_probability/pred_scope' + save_name + '.csv', index=None)

    return pred_exist_one, pred_scope


def one_to_one_factor(x_data, y_data, jk_list, model_name='rf', save_name='', number_density=100):
    print('单因子生存概率正在执行...')

    # data = pd.read_excel('./data/data_.xlsx')
    # pred_data = pd.read_excel('./pred_data/2030l.xlsx')
    # y_data = np.array(data['y'])
    # x_data = np.array(data.iloc[:, 4:])
    # x_pred_data = np.array(pred_data.iloc[:, 3:])

    y_data = np.array(y_data)
    x_data = np.array(x_data)
    x_max_min = (x_data - x_data.min(axis=0)) / (x_data.max(axis=0) - x_data.min(axis=0))
    # x_one_dat = x_max_min[y_data==1,:]
    jk_list = list(np.array(jk_list).reshape(-1))
    x_max_min = x_max_min[:, jk_list]

    pred_exist_one = {}
    pred_scope = {}
    x_data_max = x_data.max(axis=0)[jk_list]
    x_data_min = x_data.min(axis=0)[jk_list]
    for i in range(len(jk_list)):
        clf = xlf_dict[model_name]
        clf.fit(x_max_min[:, i:i + 1], y_data)
        exist_temp = np.arange(0, 0.98 + 1 / number_density, 1 / number_density)
        exist_temp = exist_temp.reshape(-1, 1)
        pred_exist_one[jk_list[i]] = (clf.predict_proba(exist_temp))[:, 1]
        pred_scope[jk_list[i]] = (x_data_max[i] - x_data_min[i]) * exist_temp.reshape(-1) + x_data_min[i]
        del clf
    pred_exist_one_pd = pd.DataFrame(pred_exist_one)
    pred_scope_pd = pd.DataFrame(pred_scope)
    if not os.path.exists('./exist_probability'):
        os.mkdir('./exist_probability')
    pred_exist_one_pd.to_csv('./exist_probability/one_to_pred_exist_one' + save_name + '.csv', index=None)
    pred_scope_pd.to_csv('./exist_probability/one_to_pred_scope' + save_name + '.csv', index=None)

    return pred_exist_one, pred_scope


# 单因子及去除分析
def factor_train(x_data, y_data, jk_list, clf_name='rf'):
    print('正在选择最优模型...')

    # data = pd.read_excel('./data/data_.xlsx')
    # y_data = np.array(data['y'])
    # x_data = np.array(data.iloc[:, 4:])
    save_file = './factor_or_no/'
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
        for tr_idx, val_idx in kfold.split(x_max_min, y_data):
            clf1 = xlf_dict[clf_name]
            clf2 = xlf_dict[clf_name]
            train_x, train_y = x_max_min[tr_idx], y_data[tr_idx]
            test_x, test_y = x_max_min[val_idx], y_data[val_idx]
            clf1.fit(train_x[:, jk_i:jk_i + 1], train_y)
            y_pred = clf1.predict(test_x[:, jk_i:jk_i + 1])
            y_score = clf1.predict_proba(test_x[:, jk_i:jk_i + 1])
            idx_all = print_data(test_y, y_pred, y_score)
            auc_dict_factor[jk_i].append(idx_all['AUROC'])

            jk_temp_list = [xx for xx in jk_list if xx != jk_i]
            clf2.fit(train_x[:, jk_temp_list], train_y)
            y_pred = clf2.predict(test_x[:, jk_temp_list])
            y_score = clf2.predict_proba(test_x[:, jk_temp_list])
            idx_all = print_data(test_y, y_pred, y_score)
            auc_dict_no_factor[jk_i].append(idx_all['AUROC'])

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


def latitude_line_block(pred_results_filename, XY=1, _block=100, label=''):
    print('正在绘制...')
    if type(pred_results_filename) == type(' '):
        pred_results_filename = [pred_results_filename]
    if type(label) == type(''):
        label = [label]
    assert len(label) == len(pred_results_filename)
    plt.figure(figsize=(20, 8), dpi=600)
    Font = {'size': 18, 'family': 'Times New Roman', 'weight': 'bold'}

    for num, pred_results_filename_i in enumerate(pred_results_filename):
        assert type(pred_results_filename_i) == type(' ')
        pred_results = pd.read_csv(pred_results_filename_i)

        assert pred_results.shape[0] > _block * 5
        assert pred_results.shape[1] == 3
        assert pred_results.iloc[:, 2].max() <= 1

        if label[num] == '':
            label[num] = pred_results.keys()[XY]

        pred_results.sort_values(pred_results.keys()[XY], inplace=True)
        plt_X = []
        plt_Y = []
        for i in range(pred_results.shape[0] - _block + 1):
            plt_Y.append(pred_results.iloc[i:i + _block, 2].sum() / _block)
            plt_X.append(pred_results.iloc[i:i + _block, XY].sum() / _block)

        plt.plot(plt_X, plt_Y, label=label[num])
    plt.legend(loc='upper right', prop=Font)
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([pred_results.min(axis=0)[XY], pred_results.max(axis=0)[XY]])
    # plt.ylim([pred_results.min(axis=0)[2], pred_results.max(axis=0)[2]])
    plt.ylabel('The survival probability', Font)
    plt.xlabel('Latitude', Font)
    # plt.tick_params(labelsize=15)
    # plt.title(name.upper())

    plt.grid()  # linewidth并不是网格宽度而是网格线的粗细
    if not os.path.exists('./img_lat_line'):
        os.mkdir('./img_lat_line')
    plt.savefig('./img_lat_line/lat_line_block.png')
    # plt.show()
    print('正在完成...')
    return 0


def latitude_line_angle(pred_results_filename, XY=1, _angle=1, label=''):  # _block=100
    print('正在绘制...')
    if type(pred_results_filename) == type(' '):
        pred_results_filename = [pred_results_filename]
    if type(label) == type(''):
        label = [label]
    assert len(label) == len(pred_results_filename)
    plt.figure(figsize=(20, 8), dpi=600)
    Font = {'size': 18, 'family': 'Times New Roman', 'weight': 'bold'}

    for num, pred_results_filename_i in enumerate(pred_results_filename):
        assert type(pred_results_filename_i) == type(' ')
        pred_results = pd.read_csv(pred_results_filename_i)

        # assert pred_results.shape[0] > _block*5
        assert pred_results.shape[1] == 3
        assert pred_results.iloc[:, 2].max() <= 1

        if label[num] == '':
            label[num] = pred_results.keys()[XY]

        pred_results.sort_values(pred_results.keys()[XY], inplace=True)
        plt_X = []
        plt_Y = []
        for i in pred_results.iloc[:, XY]:
            if i >= pred_results.iloc[:, XY].max(axis=0) - _angle:
                break
            ctl_ = (i + _angle > pred_results.iloc[:, XY]) & (pred_results.iloc[:, XY] > i)
            plt_Y.append(pred_results[ctl_].iloc[:, 2].mean())
            plt_X.append(pred_results[ctl_].iloc[:, XY].mean())

        plt.plot(plt_X, plt_Y, label=label[num])

    plt.legend(loc='upper right', prop=Font)
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([pred_results.min(axis=0)[XY], pred_results.max(axis=0)[XY]])
    # plt.ylim([pred_results.min(axis=0)[2], pred_results.max(axis=0)[2]])
    plt.ylabel('The survival probability', Font)
    plt.xlabel('Latitude', Font)
    # plt.tick_params(labelsize=15)
    # plt.title(name.upper())

    plt.grid()  # linewidth并不是网格宽度而是网格线的粗细
    if not os.path.exists('./img_lat_line'):
        os.mkdir('./img_lat_line')
    plt.savefig('./img_lat_line/lat_line_angle.png')
    # plt.show()
    print('正在完成...')
    return 0


def latitude_line_angle_block(pred_results_filename, XY=1, _angle=1, _angle_block=10000, label='',save_n=''):  # _block=100
    print('正在绘制...')
    if type(pred_results_filename) == type(' '):
        pred_results_filename = [pred_results_filename]
    if type(label) == type(''):
        label = [label]
    assert len(label) == len(pred_results_filename)
    plt.figure(figsize=(20, 8), dpi=600)
    Font = {'size': 24, 'family': 'Times New Roman', 'weight': 'bold'}

    for num, pred_results_filename_i in enumerate(pred_results_filename):
        assert type(pred_results_filename_i) == type(' ')
        pred_results = pd.read_csv(pred_results_filename_i)

        # assert pred_results.shape[0] > _block*5
        assert pred_results.shape[1] == 3
        assert pred_results.iloc[:, 2].max() <= 1

        if label[num] == '':
            label[num] = pred_results.keys()[XY]

        pred_results.sort_values(pred_results.keys()[XY], inplace=True)
        plt_X = []
        plt_Y = []
        ctl_number = (pred_results.iloc[:, XY].max(axis=0) - pred_results.iloc[:, XY].min(axis=0)) * (1 / _angle_block)
        min_res = pred_results.iloc[:, XY].min(axis=0)

        for i in range(_angle_block + 1):
            number = i * ctl_number + min_res
            number_next = _angle + number
            if number >= pred_results.iloc[:, XY].max(axis=0):
                break
            ctl_ = (number_next > pred_results.iloc[:, XY]) & (pred_results.iloc[:, XY] >= number)
            # ctl_ = (0 > pred_results.iloc[:, XY]) & (pred_results.iloc[:,XY] >= 0)
            if pd.notna(pred_results[ctl_].iloc[:, 2].mean()):
                plt_Y.append(pred_results[ctl_].iloc[:, 2].mean())
            else:
                plt_Y.append(0.0)
            plt_X.append((number_next + number) / 2)

        plt.plot(plt_X, plt_Y, label=label[num])

    plt.legend(loc='upper right', prop=Font)
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([pred_results.min(axis=0)[XY], pred_results.max(axis=0)[XY]])
    plt.ylim(-0.01, 0.8)
    plt.ylabel('The survival probability', Font)
    plt.xlabel('Latitude (°N)', Font)
    # plt.tick_params(labelsize=15)
    # plt.title(name.upper())
    ax = plt.gca()
    labs = ax.get_xticklabels()
    [label.set_fontweight('bold') for label in labs]
    [label.set_fontsize(15) for label in labs]
    labs = ax.get_yticklabels()
    [label.set_fontweight('bold') for label in labs]
    [label.set_fontsize(15) for label in labs]
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细

    plt.grid()  # linewidth并不是网格宽度而是网格线的粗细
    if not os.path.exists('./img_lat_line'):
        os.mkdir('./img_lat_line')
    plt.savefig('./img_lat_line/lat_line_a' + str(_angle) + '_b' + str(_angle_block) + save_n +'.png')
    # plt.show()
    print('正在完成...')
    return 0


def get_area_(x_data_file_name, all_area=1):
    if type(x_data_file_name) == type(' '):
        x_data_file_name = [x_data_file_name]
    if not os.path.exists('./results/area_count/'):
        os.mkdir('./results/area_count/')
    for x_data_name in x_data_file_name:
        x_data = np.array(pd.read_csv(x_data_name))
        x_data = x_data[:, -1]
        unique, counts = np.unique(x_data, return_counts=True)
        assert unique.shape[0] <= 10  # 类别应当小于10
        counts = np.expand_dims(counts / x_data.shape[0], axis=1)
        counts1 = counts * all_area
        counts = np.concatenate((counts, counts1), axis=1)
        df = pd.DataFrame(dict(zip(unique, counts)))
        name_f = x_data_name.split('/')[-1].split('.')[0]
        df.to_csv('./results/area_count/' + name_f + '.csv', index=None)
        print(df)


def grid_xlf(key, feature_i, xlf, param_dict, x_train, y_train, x_test, y_test):
    for i in param_dict.keys():
        if not (hasattr(xlf, i)):
            raise Exception('xlf_append: {} 属性在{}中不存在..'.format(i, xlf()))
    param_grid_dict = list(ParameterGrid(param_dict))

    param_EI_list = []
    for param_sigle in param_grid_dict:
        for keys in param_dict.keys():
            setattr(xlf, keys, param_sigle[keys])

        xlf.fit(x_train, y_train)
        yy_pred = xlf.predict_proba(x_test)

        y_pred = np.argmax(yy_pred, axis=1)
        true_values = np.array(y_test)
        y_scores = yy_pred[:, 1]
        TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
        Sensitivity = TP / (TP + FN)
        Specificity = TN / (TN + FP)
        ACC = (TP + TN) / (TP + FP + FN + TN)
        Precision = TP / (TP + FP)
        F1Score = 2 * TP / (2 * TP + FP + FN)
        MCC = ((TP * TN) - (FP * FN)) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
        AUC = roc_auc_score(true_values, y_scores)
        pre, rec, thresholds = precision_recall_curve(y_test, y_scores)
        prc_area = auc(rec, pre)
        EI = [key, feature_i, param_sigle, TN, FP, FN, TP, Precision, Sensitivity, Specificity, F1Score, MCC, ACC,
              prc_area, AUC]
        param_EI_list.append(EI)
    return param_EI_list


def grid_search_best_model():
    pass


if __name__ == '__main__':
    data = pd.read_excel('./data/data_.xlsx')
    pred_data = pd.read_excel('./pred_data/2030l.xlsx')
    y_data = np.array(data['y'])
    x_data = np.array(data.iloc[:, 4:])
    x_pred_data = np.array(pred_data.iloc[:, 3:])
    sites = pred_data.iloc[:, 1:3]

    x_corr = get_corr(x_data)
    jk_value = get_jickknife(x_data, y_data)
    best_feature = get_best_features(x_corr, jk_value)
    best_model = get_best_model(x_data, y_data, best_feature)
    best_model_path = './models/' + best_model + '.pkl'
    y_proba_cls = use_best_model(sites, x_pred_data, x_data, best_model_path, best_feature)
    print('结束...')
