#!/usr/bin/env python
#-*- coding: UTF-8 -*-
from data_utils import readucr, readucr2, Data_MinMax_Scaler, is_abnormal, readUcrTsv
# from GRU2 import Single_GRU, Multi_Gru,  output_of_single_gru, output_of_multi_gru
import tensorflow as tf
import kerastuner as kt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split
import os
from cnn_AE_nc_2 import Cnn_AE_nc_2
from cnn_AE_nc_3 import Cnn_AE_nc_3
from cnn_AE_nc_4 import Cnn_AE_nc_4
from cnn_AE_ns_2 import Cnn_AE_ns_2
from cnn_AE_ns_2 import Cnn_AE_ns_2
from cnn_AE_ns_3 import Cnn_AE_ns_3
from cnn_AE_ns_4 import Cnn_AE_ns_4
from cnn_AE_nproc_2 import Cnn_AE_nproc_2
from cnn_AE_nproc_3 import Cnn_AE_nproc_3
from cnn_AE_nproc_4 import Cnn_AE_nproc_4
from cnn_AE_npros_2 import Cnn_AE_npros_2
from cnn_AE_npros_3 import Cnn_AE_npros_3
from cnn_AE_npros_4 import Cnn_AE_npros_4

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# hyperparameter search
# def searchhp(model, x_train, y_train, x_val, y_val, epochs):
#     # define tuner
#     tuner = kt.Hyperband(
#         model,
#         objective='val_loss',
#         max_epochs=epochs,
#         hyperband_iterations=3)
#     # search
#     tuner.search(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, callbacks=model.set_HpCallbacks())
#     best_model = tuner.get_best_models(1)[0]
#     best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
#     return best_model, best_hyperparameters




# 相同卷积核模型, channel维度，三种层数
def train_cnn_AE_nc(input_shape, x_train, y_train, x_val, y_val, batchsize=6, epochs=400, verbose=True):
    # model = Cnn_AE_nc_2('result', input_shape, batchsize,verbose)
    # model = Cnn_AE_nc_3('result', input_shape, batchsize, verbose)
    model = Cnn_AE_nc_4('result', input_shape, batchsize, verbose)
    model.fit_model(x_train, y_train, x_val, y_val, epochs)
    # 以下语句没有用
    # x, y = searchhp(model, x_train, y_train, x_val, y_val, epochs)
    # print(x, y)



# 相同卷积核模型，sensor维度，三种层数
def train_cnn_AE_ns(input_shape, x_train, y_train, x_val, y_val, batchsize=6, epochs=400, verbose=True):
    # model = Cnn_AE_ns_2('result', input_shape, batchsize, verbose)
    # model = Cnn_AE_ns_3('result', input_shape, batchsize, verbose)
    model = Cnn_AE_ns_4('result', input_shape, batchsize, verbose)
    model.fit_model(x_train, y_train, x_val, y_val, epochs)
    # 以下语句没有用
    # x, y = searchhp(model, x_train, y_train, x_val, y_val, epochs)
    # print(x, y)



# 不同卷积核模型, channel维度，三种层数
def train_cnn_AE_nproc(input_shape, x_train, y_train, x_val, y_val, batchsize=6, epochs=400, verbose=True):
    # model = Cnn_AE_nproc_2('result', input_shape, batchsize, verbose)
    # model = Cnn_AE_nproc_3('result', input_shape, batchsize, verbose)
    model = Cnn_AE_nproc_4('result', input_shape, batchsize, verbose)
    model.fit_model(x_train, y_train, x_val, y_val, epochs)


# 不同卷积核模型，sensor维度，三种层数
def train_cnn_AE_npros(input_shape, x_train, y_train, x_val, y_val, batchsize=6, epochs=400, verbose=True):
    # model = Cnn_AE_npros_2('result', input_shape, batchsize, verbose)
    # model = Cnn_AE_npros_3('result', input_shape, batchsize, verbose)
    model = Cnn_AE_npros_4('result', input_shape, batchsize, verbose)
    model.fit_model(x_train, y_train, x_val, y_val, epochs)


if __name__ == '__main__':
    # （-1,1）之间的随机数
    mu = 0
    sigma = 1
    sample_slice = -1 + 2* np.random.random((4, 64, 20))
    # print(sample_slice[0, 0, :])
    # sample_slice_gauss = sample_slice + np.random.normal(mu, sigma, (4, 512, 20))/100
    # print(sample_slice_gauss[0, 0, :])
    # sample_slice_gauss_scaler = Data_MinMax_Scaler(np.expand_dims(sample_slice_gauss, 0))
    # print(sample_slice_gauss_scaler[0, 0, 0, :])
    # np.save('sample.npy', sample_slice)
    # sam = np.load('sample.npy')
    x_train_random = np.array([sample_slice + np.random.normal(mu, sigma, (4, 64, 20)) for i in range(1800)])
    x_val_random = np.array([sample_slice + np.random.normal(mu, sigma, (4, 64, 20)) for i in range(900)])
    # 归一化
    x_train_random = Data_MinMax_Scaler(x_train_random)
    x_val_random = Data_MinMax_Scaler(x_val_random)
    train_cnn_AE_npros((4, 64, 20), x_train_random, x_train_random, x_val_random, x_val_random, epochs=100)
    # train_cnn_AE_npron((4, 64, 20), x_train_random, x_train_random, x_val_random, x_val_random, epochs=1000)
    # train_cnn_AE_2((4, 64, 20), x_train_random, x_train_random, x_val_random, x_val_random, epochs=500)
    # cnn_AE_n_model = load_model('result/cnn_AE_n/best_model.hdf5', custom_objects={'root_mean_squared_error':root_mean_squared_error})
    # cnn_AE_npron_model = load_model('result/cnn_AE_npron/best_model.hdf5',custom_objects={'root_mean_squared_error':root_mean_squared_error})
    # test_slice = np.tile(-1 + 2* np.random.random((4, 512, 20)), (6, 1, 1, 1))
    # test_slice = np.tile(sample_slice, (6, 1, 1, 1))
    # print(test_slice[0, 0, 0, :])
    # recons_slice_n = cnn_AE_n_model.predict(test_slice)
    # recons_slice_npron = cnn_AE_npron_model.predict(test_slice)
    # print(recons_slice_n[0, 0, 0, :])
    # print(recons_slice_npron[0, 0, 0, :])





