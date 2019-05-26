#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2018/5/26 19:13
@Author : Administrator
@Email : xxxxxxxxx@qq.com
@File : LeNet_model.py
@Project : keras-demo
"""
from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense


class LeNet(object):
    """
    定义两个卷积层，一个全连接层的LeNet网络
    """

    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        model.add(Conv2D(filters=20, kernel_size=5, padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(filters=50, kernel_size=5, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # 全连接层
        model.add(Flatten())
        model.add(Dense(200))
        model.add(Activation('relu'))
        # softmax分类器
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        model.summary()

        return model





