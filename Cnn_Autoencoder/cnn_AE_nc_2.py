#!/usr/bin/env python
#-*- coding: UTF-8 -*-
import os
import time
import datetime
import tensorflow as tf
# import kerastuner as kt
# from kerastuner import HyperParameters
# from kerastuner import HyperModel
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv1D, Conv2D, BatchNormalization, \
    Input, UpSampling2D, ZeroPadding1D, ZeroPadding2D, Lambda, \
    Conv2DTranspose, Activation, Cropping2D

# tf.compat.v1.disable_eager_execution()
# os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# def abMaxPooling1D(inputs, pool_size=2, strides=2, padding='SAME'):
#     # tf.nn.max_pool(value, ksize, strides, padding, name=None)
#     # output1 = MaxPooling1D(pool_size=pool_size, strides=strides, padding=padding)(inputs)
#     # output2 = MaxPooling1D(pool_size=pool_size, strides=strides, padding=padding)(-inputs)
#     output1 = tf.nn.max_pool1d(inputs, ksize=pool_size, strides=strides, padding=padding)
#     output2 = tf.nn.max_pool1d(-inputs, ksize=pool_size, strides=strides, padding=padding)
#     mask = output1 >= output2
#     output = tf.where(mask, output1, -output2)
#     return output
#
# def abMaxPooling2D(inputs, pool_size=[2, 2], strides=2, padding='SAME'):
#     # output1 = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)(inputs)
#     # output2 = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)(-inputs)
#     output1 = tf.nn.max_pool2d(inputs, ksize=pool_size, strides=strides, padding=padding)
#     output2 = tf.nn.max_pool2d(-inputs, ksize=pool_size, strides=strides, padding=padding)
#     mask = output1 >= output2
#     output = tf.where(mask, output1, -output2)
#     return output


def abMaxPooling_with_argmax(inputs, pool_size=2, strides=2, padding='SAME'):

    output1, argmax1 = tf.nn.max_pool_with_argmax(inputs, ksize=pool_size, strides=strides, padding=padding)
    output2, argmax2 = tf.nn.max_pool_with_argmax(-inputs, ksize=pool_size, strides=strides, padding=padding)
    argmax1 = tf.stop_gradient(argmax1)
    argmax2 = tf.stop_gradient(argmax2)
    # output1 = tf.nn.max_pool2d(inputs, ksize=pool_size, strides=strides, padding=padding)
    # output2 = tf.nn.max_pool2d(-inputs, ksize=pool_size, strides=strides, padding=padding)
    mask = output1 >= output2
    output = tf.where(mask, output1, -output2)
    argmax = tf.where(mask, argmax1, argmax2)
    return (output, argmax)

def unAbMaxPooling(inputs_argmax, ksize, strides=2, padding='SAME'):
    # 假定 ksize = strides
    inputs = inputs_argmax[0]
    argmax = inputs_argmax[1]
    input_shape = inputs.get_shape()
    if padding == 'SAME':
        rows = input_shape[1] * ksize[1]
        cols = input_shape[2] * ksize[2]
    else:
        rows = (input_shape[1] - 1) * ksize[1] + ksize[1]
        cols = (input_shape[2] - 1) * ksize[2] + ksize[2]
    # 计算new shape
    output_shape = (input_shape[0], rows, cols, input_shape[3])
    # 计算索引
    one_like_mask = tf.ones_like(argmax)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = argmax // (output_shape[2] * output_shape[3])
    x = argmax % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    c = one_like_mask * feature_range
    # 转置索引
    update_size = tf.size(inputs)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, c]), [4, update_size]))
    values = tf.reshape(inputs, [update_size])
    outputs = tf.scatter_nd(indices, values, output_shape)
    return outputs


# def reshapes(x, retype):
#     if retype == 'reshapedim':
#         x = tf.expand_dims(tf.transpose(x, [0, 2, 1]), -1)
#     if retype == 'squeeze':
#         x = tf.squeeze(x, [1])
#     return x

def reshapes(x):
    x = tf.squeeze(x, [1])
    x = tf.expand_dims(tf.transpose(x, [0, 2, 1]), -1)
    return x

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


class Cnn_AE_nc_2:
    def __init__(self, output_directory, input_shape, batchsize, verbose=False, **kwargs):
        self.output_directory = output_directory
        self.input_shape = input_shape
        self.batchsize = batchsize
        self.set_Config()
        self.model = self.build_model()
        # verbose是信息展示模式
        if verbose == True:
            self.model.summary()
        self.verbose = verbose


    def set_Config(self):
        self.conv1_filters = 32
        self.conv1_kersize = 3
        self.conv2_filters = 16
        self.conv2_kersize = 3
        self.z_filters = self.conv2_filters
        self.z_kersize = 3


    def set_ModCallbacks(self):
        file_dir = os.path.join(self.output_directory, 'cnn_AE_nc_2')
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        file_path = os.path.join(file_dir, 'best_model.hdf5')

        log_dir = ".\log\\fit\\cnn_AE_nc_2\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                           monitor='val_loss',
                                                           save_best_only=True,
                                                           mode='auto')
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2,
                                                      min_lr=0.0001)

        self.callbacks = [tensorboard, model_checkpoint, reduce_lr]
        return self.callbacks


    # def set_HpCallbacks(self):
    #     reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2,
    #                                                   min_lr=0.0001)
    #     callbacks = [reduce_lr]
    #     return callbacks


    def build_model(self):
        # input --> (None, n, 576, c)
        input_layer = Input(batch_shape=(self.batchsize, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        # Encoder
        # conv block -1 （卷积+池化）
        conv1 = ZeroPadding2D((0, self.conv1_kersize//2))(input_layer)
        h1 = input_layer.get_shape()[1]
        conv1 = Conv2D(filters=self.conv1_filters, kernel_size=(h1, self.conv1_kersize))(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation(activation='relu')(conv1)
        conv1_pool, conv1_argmax = Lambda(abMaxPooling_with_argmax, arguments={'pool_size': [1, 2]}, name='abMaxPool1')(conv1)


        # conv block -2 （卷积+池化）
        conv2 = ZeroPadding2D((0, self.conv2_kersize//2))(conv1_pool)
        h2 = conv2.get_shape()[1]
        conv2 = Conv2D(filters=self.conv2_filters, kernel_size=(h2, self.conv2_kersize))(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation(activation='relu')(conv2)
        conv2_pool, conv2_argmax = Lambda(abMaxPooling_with_argmax, arguments={'pool_size': [1, 2]}, name='abMaxPool2')(conv2)

        # 中间层
        h_z = conv2_pool.get_shape()[1]
        z = ZeroPadding2D((0, self.z_kersize//2))(conv2_pool)
        z = Conv2D(filters=self.z_filters, kernel_size=(h_z, self.z_kersize))(z)
        z = BatchNormalization()(z)
        encoder = Activation(activation='relu')(z)


        # decoder
        # conv block -1 （反池化+反卷积）
        deconv1_unpool = Lambda(unAbMaxPooling, arguments={'ksize': [1, 1, 2, 1]}, name='unAbPool1')([encoder, conv2_argmax])
        deconv1 = Conv2DTranspose(filters=self.conv1_filters, kernel_size=(h2, self.conv2_kersize), padding='same')(deconv1_unpool)
        deconv1 = BatchNormalization()(deconv1)
        deconv1 = Activation(activation='relu')(deconv1)

        # conv block -2 （反池化+反卷积）
        deconv2_unpool = Lambda(unAbMaxPooling, arguments={'ksize': [1, 1, 2, 1]}, name='unAbPool2')([deconv1, conv1_argmax])
        deconv2 = Conv2DTranspose(filters=self.input_shape[2], kernel_size=(h1, self.conv1_kersize), padding='valid')(deconv2_unpool)
        deconv2 = Cropping2D(cropping=((0, 0), (self.conv1_kersize//2, self.conv1_kersize//2)))(deconv2)
        deconv2 = BatchNormalization()(deconv2)
        output_layer = Activation(activation='tanh')(deconv2)



        model = Model(inputs=input_layer, outputs=output_layer)
        # hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        model.compile(loss=root_mean_squared_error,
                      optimizer=optimizers.Adam(0.001),
                      metrics=[root_mean_squared_error],
                      experimental_run_tf_function=False)

        return model

    def fit_model(self, x_train, y_train, x_val, y_val, epochs):
        # x_val and y_val are only used to monitor the test loss and NOT for training
        # batch_size = 12
        self.set_ModCallbacks()
        nb_epochs = epochs

        # 小批量训练大小
        # mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))
        batch_size = 6

        # 开始时间
        start_time = time.time()

        # file_path = os.path.join(self.output_directory, 'cnn_AE_n/best_model.hdf5')

        # 训练模型
        hist = self.model.fit(x_train, y_train,
                              epochs=nb_epochs,
                              verbose=self.verbose,
                              validation_data=(x_val, y_val),
                              callbacks=self.callbacks)

        # print('history：', hist.history)
        # 训练持续时间
        duration = time.time() - start_time
        print('duration: ', duration)
        keras.backend.clear_session()

        # 做测试，所以需要加载模型
        # model = load_model(file_path)
        # loss = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
        # print('test_loss: ', loss)
        # save_logs(self.output_directory, hist, y_pred, y_true, duration, lr=False)

