import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from collections import Counter

# 2, 3, 24, 30, 34, 38, 67
# 34有0值，需要做处理，67有0值，需要做处理
# for i in [3]:
#     # time_series_df = pd.read_csv('data/Yahoo_S5/A1Benchmark/real_'+str(i)+'.csv')
#     time_series_df = pd.read_csv('data/Yahoo_S5/dataset/real_'+str(i)+'_10%.csv')
#     dataset_value = pd.DataFrame()
#     dataset_value['value'] = time_series_df.value.values
#     dataset_value.plot()
#     # x = np.where(time_series_df.is_anomaly.values == 1)
#     # y = [dataset_value['value'][index] for index in x]
#     # plt.scatter(x, y, s=25, c='r')
#     plt.title('real_'+str(i)+'.csv')
#     plt.show()

# 读取数据集
# time_series_df = pd.read_csv('data/NAB/ec2_cpu_utilization_53ea38.csv')
# time_series_df = pd.read_csv('data/NAB/realKnownCause/nyc_taxi.csv')
# dataset_value = pd.DataFrame()
# dataset_value['value'] = time_series_df.value.values[:700]
# dataset_value.plot()
# plt.title('nyc_taxi')
# plt.show()
# # print(time_series_df.head())
# dataset_value = time_series_df.value.values
# # dataset_labels = time_series_df.is_anomaly.values
# actual_vals = dataset_value
# # Counter(dataset_labels)
# # print(sum(dataset_labels == 1))
# #
# train, test = actual_vals[:300], actual_vals[300:]
# train_log, test_log = np.log10(train), np.log10(test)
# # actual_log = np.log10(actual_vals)
#
# decomposition = seasonal_decompose(dataset_value['value'], period=48, two_sided=False)
# # self.ts:时间序列，series类型;
# # freq:周期，
# # two_sided:观察下图2、4行图，左边空了一段，如果设为True，则会出现左右两边都空出来的情况，False保证序列在最后的时间也有数据，方便预测。
# trend = decomposition.trend
# seasonal = decomposition.seasonal
# residual = decomposition.resid
# decomposition.plot()
# plt.title('nyc_taxi')
# plt.show()

time_series_df = pd.read_excel('result/real38_predicted_10%.xlsx')
print(time_series_df.head())
# figsize = (12, 7)
# plt.figure(figsize=figsize)
# plt.plot(time_series_df.predicted.values, label='predicted')
# plt.savefig("figure/test.png")
# plot
actuals = time_series_df.actuals.values
predicted = time_series_df.predicted.values.astype(np.float32)
figsize = (12, 7)
plt.figure(figsize=figsize)
plt.plot(actuals, label='Actuals')
plt.plot(predicted, color='red', label='Predicted')
plt.legend(loc='upper left')
plt.title("SARIMA for real_38_10%")
# 保存图片
plt.savefig("figure/real38_predicted_10%.png")
plt.show()
