#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import data_utils as daut
import os
import warnings
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller  # 导入ADF检验函数
from statsmodels.tsa.seasonal import seasonal_decompose  # 导入季节性分解函数，将数列分解为趋势、季节性和残差三部分
from statsmodels.stats.diagnostic import acorr_ljungbox  # 导入白噪声检验函数
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf  # 导入自相关和偏自相关的绘图函数
from matplotlib.ticker import MaxNLocator  # 导入自动查找到最佳的最大刻度函数
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn import svm
from pmdarima import auto_arima

warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
# import statsmodels.tsa.api as smt
from sklearn.metrics import mean_squared_error


# root mean squared error or rmse
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
    df = df.sort_values(by='load_date', ascending=False)
    df.load_date = pd.to_datetime(df['load_date'].astype(str), format="%Y-%m-%d")
    return df


time_series_df = pd.read_csv('data/Yahoo_S5/dataset/real_2_5%.csv')
print(time_series_df.head())
dataset_value = time_series_df.value.values
dataset_labels = time_series_df.is_anomaly.values
dataset_timestamp = time_series_df.timestamp.values
actual_vals = dataset_value
# actual_log = np.log10(actual_vals)

# plot原数据
# dataset_df = pd.DataFrame()
# dataset_df['value'] = dataset_value[-200:-100]
# dataset_df.plot()
# plt.show()

train, test = actual_vals[:700], actual_vals[700:]
test_labels = dataset_labels[700:]
test_anomaly_index = np.where(test_labels == 1)[0]
train_log, test_log = np.log10(train), np.log10(test)
stepwise_model = auto_arima(train_log, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=24,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
# my_order = (1, 1, 1)
# my_seasonal_order = (0, 1, 1, 24)
print(stepwise_model.summary())
# history = [x for x in train_log]
# predictions = list()
# predict_log = list()
# for t in range(len(test_log)):
#     model = sm.tsa.SARIMAX(history, order=my_order, seasonal_order=my_seasonal_order,
#                            enforce_stationarity=False, enforce_invertibility=False)
#     model_fit = model.fit(disp=0)
#     output = model_fit.forecast()
#     # stepwise_model.fit(history)
#     # output = stepwise_model.predict(n_periods=1)
#     predict_log.append(output[0])
#     y_hat = 10 ** output[0]
#     predictions.append(y_hat)
#     obs = test_log[t]
#     history.append(obs)
#     print('predicted=%f, expected=%f' % (output[0], obs))
# error = measure_rmse(test_log, predict_log)
# print('Test rmse: %.3f' % error)
#
# predicted_df=pd.DataFrame()
# predicted_df['actuals'] = test
# predicted_df['predicted'] = predictions
# predicted_df['actuals_labels'] = test_labels
# predicted_df['timestamp'] = dataset_timestamp[700:]
# print(predicted_df.head())
# # 保存预测数据
# writer = pd.ExcelWriter("result/real24_predicted_5%.xlsx", encoding="utf-8-sig")
# predicted_df.to_excel(writer, "sheet1")
# writer.save()
# print("数据保存成功")
#
# # plot
# figsize = (12, 7)
# plt.figure(figsize=figsize)
# plt.plot(test, label='Actuals')
# plt.plot(predictions, color='red', label='Predicted')
# plt.legend(loc='upper left')
# plt.title("SARIMA for real_24_5%")
# # 保存图片
# plt.savefig("figure/real24_predicted_5%.png")
# plt.show()


# 单位根检验(test_stationarity，ADF检验)，用于检验序列是否是平稳的
# 季节性分解函数(seasonal_decompose)，通过分解后的趋势、季节性确认序列是否是平稳的
# 白噪声检验函数
# 自相关性和偏自相关性(acf_pacf)，通过截尾或拖尾的lag值，初步确认p,q。也可以用来检验序列是否平稳

# ADF检验：这是一种检查数据稳定性的统计测试。无效假设：时间序列是不稳定的。
# 测试结果由测试统计量和一些置信区间的临界值组成。
# 如果“测试统计量”少于“临界值”，我们可以拒绝无效假设，并认为序列是稳定的。
# 当p-value<0.05，且Test Statistic显著小于Critical Value (5%)时，数列稳定
# 主要看p-value，显著小于的判断不精确
def test_stationarity(timeseries, window=12):
    rolmean = timeseries.rolling(window=window, center=False).mean()
    rolstd = timeseries.rolling(window=window, center=False).std()
    # 旧版方法，即将被移除
    # rolmean = pd.rolling_mean(timeseries, window=window)
    # rolstd = pd.rolling_std(timeseries, window=window)
    # 设置原始图，移动平均图和标准差图的式样
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')  # 使用自动最佳的图例显示位置
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    print('ADF检验结果：')
    dftest = adfuller(timeseries, autolag='AIC')  # 使用减小AIC的办法估算ADF测试所需的滞后数
    # 将ADF测试结果、显著性概率、所用的滞后数和所用的观测数打印出来
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', 'Num Lags Used', 'Num Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


# 按照滑动均值将维经过指数转化的时间序列分为趋势(trend), 季节性(seasonality)和残差(residual)三部分
def decompose_plot(series, title=''):
    decomposition = seasonal_decompose(series)  # 季节性分解函数
    trend = decomposition.trend  # 分解出的趋势，包含NaN值，原因不明
    seasonal = decomposition.seasonal  # 分解出的季节性
    residual = decomposition.resid  # 分解出的残差，包含NaN值，原因不明
    fig = decomposition.plot()
    fig.set_size_inches(12, 6)
    fig.suptitle(title)
    fig.tight_layout()
    fig2 = acf_pacf(residual, title='Residuals', figsize=(12, 6))  # 分析残差的自相关，偏自相关
    # test_stationarity(residual.dropna())  # Dropna后才能做稳定性检验
    # 原数据的残差一般是不稳定的，经过差分后的数据残差可能是平稳的


# 定义一个画自相关，偏自相关的函数
# series 输入的时间序列
# lags 自相关和偏自相关函数的滞后取值范围
def acf_pacf(series, lags=40, title=None, figsize=(12, 6)):
    # 求自相关函数
    fig = plt.figure(figsize=figsize)  # figure指设置图形的特征。figsize为设置图形大小，单位为inch
    ax1 = fig.add_subplot(211)  # 子图2行1列的第一张,大于10后用逗号分隔，如(3,4,10)
    ax1.set_xlabel('lags')  # 横坐标为滞后值
    ax1.set_ylabel('AutoCorrelation')  # 纵坐标为自相关系数
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # 设置主坐标轴为自动设置最佳的整数型坐标标签
    plot_acf(series.dropna(), lags=lags, ax=ax1)  # 没有title参数，需要删除title
    # 求偏自相关函数
    ax2 = fig.add_subplot(212)
    ax2.set_xlabel('lags')
    ax2.set_ylabel('Partial AutoCorrelation')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    plot_pacf(series.dropna(), lags=lags, ax=ax2)  # 没有title参数，需要删除title
    plt.tight_layout()  # 设置为紧凑布局
    plt.show()


# #生成一个伪随机白噪声用于测试acorr_ljungbox的可靠性
# from random import gauss
# from random import seed
# from pandas import Series
# # seed random number generator
# # seed(1) #指定生成伪随机数的种子，指定后每次生成的随机数相同
# # create white noise series
# whitenoise = Series([gauss(0.0, 1.0) for i in range(1000)])#创建一个高斯分布的白噪声
# acf_pacf(whitenoise, lags=40, title='White Noise Series')
# print(u'白噪声检验结果为：', acorr_ljungbox(whitenoise, lags=1))#检验结果：平稳度，p-value。p-value>0.05为白噪声

# # 骑自行车人数预测
# df = pd.read_csv('D:\\portland-oregon-average-monthly-.csv')
# print(df.head(), '\nindex type:\n', type(df.index))
# df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
# # 索引并resample为月
# indexed_df = df.set_index('Month')
# ts = indexed_df['riders']
# ts = ts.resample('M').sum()
# # ts.plot(title='Monthly Num. of Ridership')
# # print(ts.head(), '\nindex type:\n', type(ts.index))
#
# # #原始数据分析
# # test_stationarity(ts)
# # decompose_plot(ts, title='ts decompose')
# # print(u'白噪声检验结果为：', acorr_ljungbox(ts, lags=1))# 是否为白噪声测试,p-value<0.05非白噪声。平稳且非白噪声函数，可用于时间序列建模
# # acf_pacf(ts, lags=20, title='ts ACF&PACF')# 自相关和偏自相关初步确认p,q
# # 季节差分12个月
# ts_sdiff = ts.diff(12)
# test_stationarity(ts_sdiff.dropna())
# decompose_plot(ts_sdiff.dropna(), title='ts_sdiff decompose')
# print(u'白噪声检验结果为：', acorr_ljungbox(ts_sdiff.dropna(), lags=1))  # 是否为白噪声测试,p-value<0.05非白噪声。平稳且非白噪声函数，可用于时间序列建模
# acf_pacf(ts_sdiff.dropna(), lags=20, title='ts_sdiff ACF&PACF')  # 自相关和偏自相关初步确认p,q
# # 趋势差分1个月
# ts_diff_of_sdiff = ts_sdiff.diff(1)
# test_stationarity(ts_diff_of_sdiff.dropna())
# decompose_plot(ts_diff_of_sdiff.dropna(), title='ts_diff_of_sdiff decompose')
# print(u'白噪声检验结果为：', acorr_ljungbox(ts_diff_of_sdiff.dropna(), lags=1))  # 是否为白噪声测试,p-value<0.05非白噪声。平稳且非白噪声函数，可用于时间序列建模
# acf_pacf(ts_diff_of_sdiff.dropna(), lags=20, title='ts_diff_of_sdiff ACF&PACF')  # 自相关和偏自相关初步确认p,q
# # 得出ACF&PACF后，开始计算使用的p,q,P,Q
# # PACF(lag=k)=0，则p=k-1。如果PACF=0时，ACF仍然显著>0，则再增加一些p
# # ACF(lag=k)=0,则q=k-1。如果ACF=0时，PACF仍然显著>0，则再增加一些q
# # P: 如果季节差分后序列的ACF(lag=季节周期)显著为正, 考虑增加P
# # Q: 如果季节差分后序列的ACF(lag=季节周期)显著为负, 考虑增加Q
# # 绘出拟合后的ARIMA的拟合残差的ACF和PACF, 按上述规则调整p, q, P, Q
# # 通常P, Q最多为1
# # 这里考虑p=0, q=0，季节周期=12, 季节差分后的ACF(12)显著为负, 可以考虑P=0, Q=1
# pdq = (5, 1, 0)
# PDQ = (0, 1, 1, 12)
# # SARIMA的参数: order = p, d, q, seasonal_order=P, D, Q, season的周期=12
# model = SARIMAX(ts, order=pdq,seasonal_order=PDQ).fit()# 已拟合的模型
# print(model.summary())
