import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error

def measure_rmse(actual, predicted):
    return math.sqrt(mean_squared_error(actual, predicted))


def detect_classify_anomalies(df, window):
    df.replace([np.inf, -np.inf], np.NaN, inplace=True)
    df.fillna(0, inplace=True)
    df['error'] = df['actuals'] - df['predicted']
    df['percentage_change'] = ((df['actuals'] - df['predicted']) / df['actuals']) * 100
    df['meanval'] = df['error'].rolling(window=window).mean()
    df['deviation'] = df['error'].rolling(window=window).std()
    df['-3s'] = df['meanval'] - (2 * df['deviation'])
    df['3s'] = df['meanval'] + (2 * df['deviation'])
    df['-2s'] = df['meanval'] - (1.75 * df['deviation'])
    df['2s'] = df['meanval'] + (1.75 * df['deviation'])
    df['-1s'] = df['meanval'] - (1.5 * df['deviation'])
    df['1s'] = df['meanval'] + (1.5 * df['deviation'])
    cut_list = df[['error', '-3s', '-2s', '-1s', 'meanval', '1s', '2s', '3s']]
    cut_values = cut_list.values
    cut_sort = np.sort(cut_values)
    df['impact'] = [(lambda x: np.where(cut_sort == df['error'][x])[1][0])(x) for x in
                    range(len(df['error']))]
    severity = {0: 3, 1: 2, 2: 1, 3: 0, 4: 0, 5: 1, 6: 2, 7: 3}
    region = {0: "NEGATIVE", 1: "NEGATIVE", 2: "NEGATIVE", 3: "NEGATIVE", 4: "POSITIVE", 5: "POSITIVE", 6: "POSITIVE",
              7: "POSITIVE"}
    df['color'] = df['impact'].map(severity)
    df['region'] = df['impact'].map(region)
    df['anomaly_points'] = np.where(df['color'] == 3, df['error'], np.nan)
    df = df.sort_values(by='timestamp', ascending=True)
    # df.load_date = pd.to_datetime(df['load_date'].astype(str), format="%Y-%m-%d")
    return df


# predicted_df = pd.read_excel(r'result/real2_predicted_10%.xlsx', sheet_name='sheet1')
predicted_df = pd.read_excel(r'result/real3_predicted_5%.xlsx', sheet_name='sheet1')
print(predicted_df.head())
# 实际值
actual_values = predicted_df.actuals.values
# 预测值
predicted_values = predicted_df.predicted.values.astype(np.float32)
# 实际标签
actual_labels = predicted_df.actuals_labels.values
# 阈值
# argmax = np.where(np.isinf(predicted_values))
# actual_values1 = np.append(actual_values[:303],actual_values[304:500])
# predicted_values1 = np.append(predicted_values[:303], predicted_values[304:500])
# threshold = 1.2 * (measure_rmse(actual_values1, predicted_values1))
threshold = 0.3 * (measure_rmse(actual_values[:500],predicted_values[:500]))
dif = np.sqrt(np.square((actual_values-predicted_values)))
# 使用3sigma检验
# anomaly_classify_df = detect_classify_anomalies(predicted_df, 7)
# anomaly_classify_df = pd.read_excel(r'result/real2_classify.xlsx')
# predicted_labels = np.where(np.isnan(anomaly_classify_df['anomaly_points']), 0, 1)
# 预测标签
predicted_labels = np.where(dif > threshold, 1, 0)
# 实际异常值索引
actual_anomaly_index = np.where(actual_labels == 1)[0]
# 实际异常值索引
actual_normal_index = np.where(actual_labels == 0)[0]
# 预测异常值索引
pred_anomaly_index = np.where(predicted_labels == 1)[0]
# 预测正常值索引
pred_normal_index = np.where(predicted_labels == 0)[0]
TP = len(np.intersect1d(actual_anomaly_index, pred_anomaly_index))
TN = len(np.intersect1d(actual_normal_index, pred_normal_index))
FN = len(np.intersect1d(actual_anomaly_index, pred_normal_index))
accuracy = (TP+TN)/len(actual_values)
if len(pred_anomaly_index) == 0:
    precision = 0
else:
    precision = TP / len(pred_anomaly_index)
if (TP + FN) == 0:
    recall = 0
else:
    recall = TP / (TP + FN)
if precision == 0 or recall == 0:
    F1_Score = 0
else:
    F1_Score = 2 / (1 / precision + 1 / recall)
# precision = TP/len(pred_anomaly_index)
# recall = TP/(TP+FN)
# F1_Score = 2 / (1/precision + 1/recall)
print("数据集:", 'real_3_5%')
print("模型:", 'SARIMA')
print("Accuracy:", '%.2f%%' % (accuracy * 100))
print("Precision:", '%.2f%%' % (precision * 100))
print("Recall:", '%.2f%%' % (recall * 100))
print("F1-Score:", '%.2f%%' % (F1_Score * 100))
# print("数据集:", 'real_3')
# print("模型:", 'SARIMA')
# print("Accuracy for real3:", '%.2f%%' % (accuracy * 100))
# print("Precision for real3:", '%.2f%%' % (precision * 100))
# print("Recall for real3:", '%.2f%%' % (recall * 100))
# print("F1-Score for real3:", '%.2f%%' % (F1_Score * 100))

