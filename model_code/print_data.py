#!/opt/share/bin/anaconda3/bin python
# coding: utf-8
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve,auc
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")
import os
import sys

file_name_init = (str(os.path.basename(sys.argv[0])).split('.'))[0]  # 获取本文件（执行文件）的名字
# y_scores1 = xlf.predict_proba(X_test_new)

def my_score(y_true,y_pred):
    TN,FP,FN,TP = metrics.confusion_matrix(y_true, y_pred).ravel()
    return TN,FP,FN,TP


def print_data(y_test, y_scores, y_scores1,name_add = '.'):
    if name_add == '.':
        file_name = file_name_init
    else:
        file_name = file_name_init + '_' + name_add + '_'
    TN, FP, FN, TP = confusion_matrix(y_test, y_scores).ravel()
    Sensitivity=TP/(TP+FN)
    Specificity=TN/(TN+FP)
    ACC = (TP + TN) / (TP + FP + FN + TN)
    Precision=TP/(TP+FP)
    F1Score=2*TP/(2*TP+FP+FN)
    MCC = metrics.matthews_corrcoef(y_test, y_scores)
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores1[:,1])

    area = auc(recall, precision)

    fpr, tpr, thresholds = roc_curve(y_test, y_scores1[:,1])
    area1 = auc(fpr,tpr)
    # print("Recall:{:.3f}".format(Sensitivity))
    # print("Specificity:{:.3f}".format(Specificity))
    # print("ACCURACY:{:.3f}".format(ACC))
    # print("F1Score:{:.3f}".format(F1Score))
    # print("AUPRC for test is:{:.3f}".format(area))
    # print("AUROC for test is:{:.3f}".format(area1))
    recall = TP / (TP + FN)
    fpr = FP / (FP+TN)
    FOR = FN / (FN + TN)
    pre = TP / (TP+FP)
    # return_data = {'recall': Sensitivity, 'Specificity': Specificity, "ACC": ACC, 'F1Score': F1Score, 'AUPRC': area, 'AUROC_1': area1}
    return_data = {'TN':TN, 'FP':FP, 'FN':FN, 'TP':TP, 'F1':F1Score, 'RECALL':recall, 'SPE':Specificity, 'SEN':Sensitivity, 'ACC':ACC, 'MCC':MCC,'AUPRC':area, 'AUROC':area1,'FPR':fpr,'FOR':FOR,"PRE":pre }
    # return_data_root = {}
    # return_data_root['best_idp_scores'] = return_data # 这里只是为了符合格式才这样命名的！！！！！

    return return_data

