#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
from pyod.models.lof import LOF
import matplotlib.pyplot as plt
import numpy as np


clf_name = 'LOF'
print("模型:", 'LOF')
for i in [2, 3, 24, 30, 34, 38, 67]:
    # ts_test_df = pd.read_csv('data/Yahoo_S5/dataset/real_2_'+str(i)+'%.csv')
    # ts_test_df = pd.read_csv('data/Yahoo_S5/A1Benchmark/real_'+str(i)+'.csv')
    # ts_test_df = pd.read_csv('data/Yahoo_S5/dataset/real_'+str(i)+'_'+'5%.csv')
    ts_test_df = pd.read_csv('data/Yahoo_S5/dataset/real_'+str(i)+'_'+'10%.csv')
    # ts_test_df = pd.read_csv('data/Yahoo_S5/dataset/real_'+str(i)+'_'+'15%.csv')
    ts_test_df.head()
    test_dataset_value = ts_test_df.value.values[700:]
    test_dataset_labels = ts_test_df.is_anomaly.values[700:]
    test_dataset_timestamp = ts_test_df.timestamp.values[700:]
    X_test = test_dataset_value.reshape((-1, 1))
    y_test = test_dataset_labels.reshape((-1, 1))
    # train COF detector
    contamination = 0.1
    n_neighbors = 20
    clf = LOF(n_neighbors=n_neighbors, contamination=contamination)
    clf.fit(X_test)
    # get the prediction labels and outlier scores of the test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    # y_test_scores = clf.decision_function(X_test)  # outlier scores
    # 测试集
    # 实际异常值索引
    test_actual_anomaly_index = np.where(test_dataset_labels == 1)[0]
    # 实际正常值索引
    test_actual_normal_index = np.where(test_dataset_labels == 0)[0]
    # 预测异常值索引
    test_pred_anomaly_index = np.where(y_test_pred == 1)[0]
    # 预测正常值索引
    test_pred_normal_index = np.where(y_test_pred == 0)[0]
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
    print("数据集:", 'real_' + str(i) + '_' + '10%.csv')
    # print("数据集:", 'real_' + str(i) + '_' + '15%.csv')
    print("contamination：", contamination, "  n_neighbors：", n_neighbors)
    print("Accuracy:", '%.2f%%' % (test_accuracy * 100))
    print("Precision:", '%.2f%%' % (test_precision * 100))
    print("Recall:", '%.2f%%' % (test_recall * 100))
    print("F1-Score:", '%.2f%%' % (test_F1_Score * 100))