#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from pmdarima import auto_arima
import openpyxl

def measure_rmse(actual, predicted):
    return math.sqrt(mean_squared_error(actual, predicted))


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    # acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae,
            'mpe': mpe, 'rmse':rmse,
            'corr':corr, 'minmax':minmax})


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


# time_series_df = pd.read_csv('data/NAB/artificialNoAnomaly/art_noisy.csv')
# time_series_df.head()
# 读取数据集
time_series_df = pd.read_csv('data/Yahoo_S5/dataset/real_67_5%.csv')
print(time_series_df.head())
dataset_value = time_series_df.value.values
dataset_labels = time_series_df.is_anomaly.values
dataset_timestamp = time_series_df.timestamp.values
actual_vals = dataset_value
# actual_log = np.log10(actual_vals)
# ts_test_df = pd.read_csv('data/Yahoo_S5/A1Benchmark/real_3.csv')
# ts_test_df.head()
# test_dataset_value = ts_test_df.value.values
# test_dataset_labels = ts_test_df.is_anomaly.values
# X_test = test_dataset_value.reshape((-1, 1))
# y_test = test_dataset_labels.reshape((-1, 1))
# time_series_df.load_date = pd.to_datetime(time_series_df.load_date, format='%Y%m%d')
# time_series_df.head()

# plot原数据
# dataset_df = pd.DataFrame()
# dataset_df['value'] = dataset_value[-200:-100]
# dataset_df.plot()
# x = np.where(dataset_labels == 1)
# y = [dataset_value[index] for index in x]
# plt.scatter(x, y, s=25, c='r')
# plt.show()

# 观察自相关系数
# autocorrelation_plot(dataset_df)
# plt.show()
train, test = actual_vals[:700], actual_vals[700:]
test_labels = dataset_labels[700:]
test_anomaly_index = np.where(test_labels == 1)[0]
train_log, test_log = np.log10(train+1), np.log10(test+1)

stepwise_model = auto_arima(train_log, start_p=1, start_q=1, test='adf',
                            max_p=5, max_q=3, m=1,
                            start_P=0, seasonal=False,
                            d=1, D=1, trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)
print(stepwise_model.summary())
# model = ARIMA(history, order=(10, 1, 0))
history = [x for x in train_log]
predictions = list()
predict_log = list()
for t in range(len(test_log)):
    # model_fit = model.fit(disp=0)
    # output = model_fit.forecast()
    stepwise_model.fit(history)
    output = stepwise_model.predict(n_periods=1)
    predict_log.append(output[0])
    y_hat = 10 ** output[0] - 1
    predictions.append(y_hat)
    obs = test_log[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (output[0], test_log[t]))
error = measure_rmse(test_log, predict_log)
print('Test rmse: %.3f' % error)
# plot
figsize = (12, 7)
plt.figure(figsize=figsize)
plt.plot(test, label='Actuals')
plt.plot(predictions, color='red', label='Predicted')
x = test_anomaly_index
y = [test[index] for index in x]
plt.scatter(x, y, s=25, c='green')
plt.legend(loc='upper left')
plt.title("ARIMA for real_67_5%")
# 保存图片
plt.savefig("figure/real67_predicted_5%.png")
plt.show()

predicted_df = pd.DataFrame()
predicted_df['actuals'] = test
predicted_df['predicted'] = predictions
predicted_df['actuals_labels'] = test_labels
predicted_df['timestamp'] = dataset_timestamp[700:]
print(predicted_df.head())
# 保存预测数据
writer = pd.ExcelWriter("result/real67_predicted_5%.xlsx", encoding="utf-8-sig")
predicted_df.to_excel(writer, "sheet1")
writer.save()
print("数据保存成功")
# 保存分类结果
# detect_anomaly_df = detect_classify_anomalies(predicted_df, 7)
# writer = pd.ExcelWriter("result/real2_classify.xlsx", encoding="utf-8-sig")
# detect_anomaly_df.to_excel(writer, "sheet1")
# writer.save()
# print("数据保存成功")
