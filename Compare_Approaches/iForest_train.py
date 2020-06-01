#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy import stats

rng = np.random.RandomState(42)

# 构造训练样本
# n_samples = 200  # 样本总数
# outliers_fraction = 0.25  # 异常样本比例
# n_inliers = int((1. - outliers_fraction) * n_samples)
# n_outliers = int(outliers_fraction * n_samples)
#
# X = 0.3 * rng.randn(n_inliers // 2, 2)
# X_train = np.r_[X + 2, X - 2]  # 正常样本
# X_train = np.r_[X_train, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]  # 正常样本加上异常样本

# ts_train_df = pd.read_csv('data/Yahoo_S5/A1Benchmark/real_2.csv')
# ts_train_df.head()
# train_dataset_value = ts_train_df.value.values[700:]
# train_dataset_labels = ts_train_df.is_anomaly.values[700:]
# train_dataset_timestamp = ts_train_df.timestamp.values[700:]
# X_train = train_dataset_value.reshape((-1, 1))
# y_train = train_dataset_labels.reshape((-1, 1))
#
# # fit the model
clf_name = 'isolationForest'
# clf = IsolationForest(max_samples=len(X_train), random_state=rng, contamination=0.01)
# clf.fit(X_train)
# y_train_pred = clf.predict(X_train)  # -1是异常点
# # 训练集
# # 实际异常值索引
# train_actual_anomaly_index = np.where(train_dataset_labels == 1)[0]
# # 实际正常值索引
# train_actual_normal_index = np.where(train_dataset_labels == 0)[0]
# # 预测异常值索引
# train_pred_anomaly_index = np.where(y_train_pred == 1)[0]
# # 预测正常值索引
# train_pred_normal_index = np.where(y_train_pred == 0)[0]
# train_TP = len(np.intersect1d(train_actual_anomaly_index, train_pred_anomaly_index))
# train_TN = len(np.intersect1d(train_actual_normal_index, train_pred_normal_index))
# train_FN = len(np.intersect1d(train_actual_anomaly_index, train_pred_normal_index))
# train_accuracy = (train_TP+train_TN)/len(train_dataset_labels)
# train_precision = train_TP/len(train_pred_anomaly_index)
# train_recall = train_TP/(train_TP+train_FN)
# train_F1_Score = 2 / (1/train_precision + 1/train_recall)
print("模型:", 'isolationForest')
# print("数据集:", 'real_2_2%')
# print("Accuracy:", '%.2f%%' % (train_accuracy * 100))
# print("Precision:", '%.2f%%' % (train_precision * 100))
# print("Recall:", '%.2f%%' % (train_recall * 100))
# print("F1-Score:", '%.2f%%' % (train_F1_Score * 100))

for i in [2, 3, 24, 30, 34, 38, 67]:
    # ts_test_df = pd.read_csv('data/Yahoo_S5/A1Benchmark/real_'+str(i)+'.csv')
    # ts_test_df = pd.read_csv('data/Yahoo_S5/dataset/real_2_'+str(i)+'%.csv')
    # ts_test_df = pd.read_csv('data/Yahoo_S5/dataset/real_'+str(i)+'_'+'5%.csv')
    # ts_test_df = pd.read_csv('data/Yahoo_S5/dataset/real_'+str(i)+'_'+'10%.csv')
    ts_test_df = pd.read_csv('data/Yahoo_S5/dataset/real_'+str(i)+'_'+'15%.csv')
    ts_test_df.head()
    test_dataset_value = ts_test_df.value.values[700:]
    test_dataset_labels = ts_test_df.is_anomaly.values[700:]
    test_dataset_timestamp = ts_test_df.timestamp.values[700:]
    X_test = test_dataset_value.reshape((-1, 1))
    y_test = test_dataset_labels.reshape((-1, 1))
    # train COF detector
    contamination = 0.14
    clf = IsolationForest(max_samples=len(X_test), random_state=rng, contamination=contamination)
    clf.fit(X_test)
    y_test_pred = clf.predict(X_test)  # -1是异常点， 1是正常点
    # y_test_scores = clf.decision_function(X_test)  # outlier scores
    # 测试集
    # 实际异常值索引
    test_actual_anomaly_index = np.where(test_dataset_labels == 1)[0]
    # 实际正常值索引
    test_actual_normal_index = np.where(test_dataset_labels == 0)[0]
    # 预测异常值索引
    test_pred_anomaly_index = np.where(y_test_pred == -1)[0]
    # 预测正常值索引
    test_pred_normal_index = np.where(y_test_pred == 1)[0]
    test_TP = len(np.intersect1d(test_actual_anomaly_index, test_pred_anomaly_index))
    test_TN = len(np.intersect1d(test_actual_normal_index, test_pred_normal_index))
    test_FN = len(np.intersect1d(test_actual_anomaly_index, test_pred_normal_index))
    test_accuracy = (test_TP + test_TN) / len(test_dataset_labels)
    if len(test_pred_anomaly_index) ==0:
        test_precision = 0
    else:
        test_precision = test_TP / len(test_pred_anomaly_index)
    if (test_TP + test_FN)==0:
        test_recall = 0
    else:
        test_recall = test_TP / (test_TP + test_FN)
    if test_precision ==0 or test_recall ==0:
        test_F1_Score = 0
    else:
        test_F1_Score = 2 / (1 / test_precision + 1 / test_recall)
    # print("数据集:", 'real_2_'+str(i)+'%')
    # print("数据集:", 'real_' + str(i) + '_'+'2%.csv')
    # print("数据集:", 'real_' + str(i) + '_'+'5%.csv')
    # print("数据集:", 'real_' + str(i) + '_' + '10%.csv')
    print("数据集:", 'real_' + str(i) + '_' + '15%.csv')
    print("contamination：", contamination)
    print("Accuracy:", '%.2f%%' % (test_accuracy * 100))
    print("Precision:", '%.2f%%' % (test_precision * 100))
    print("Recall:", '%.2f%%' % (test_recall * 100))
    print("F1-Score:", '%.2f%%' % (test_F1_Score * 100))





