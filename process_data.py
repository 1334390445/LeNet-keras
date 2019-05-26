#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2018/5/26 19:39
@Author : Administrator
@Email : xxxxxxxxx@qq.com
@File : process_data.py
@Project : keras-demo
"""

from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as k
import numpy as np


def load_data():
    # 加载数据
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    k.set_image_dim_ordering("th")
    print(X_train.shape)
    return (X_train, y_train), (X_test, y_test)


def process_data(X_train, y_train, X_test, y_test):
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # 归一化
    X_train /= 255
    X_test /= 255

    # 训练数据和测试数据的输入格式
    X_train = X_train[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]

    print(X_train.shape, 'train samples')
    print(X_test.shape, 'test samples')

    # 将类别转为二值类别矩阵
    NB_CLASSES = 10
    Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

    return X_train, Y_train, X_test, Y_test
