#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2018/5/26 19:53
@Author : Administrator
@Email : xxxxxxxxx@qq.com
@File : train.py
@Project : keras-demo
"""
from __future__ import print_function
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
from LeNet_model import LeNet
from process_data import load_data, process_data
import os, sys

import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 定义参数

    NB_CLASSES = 10
    BATCH_SIZE = 128
    VERBOSE = 1
    OPTIMIZER = Adam()
    VALIDATION_SPLIT = 0.2
    IMG_ROWS, IMG_COLS = 28, 28
    INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS)
    NB_EPOCHS = 50
    MODEL_PATH = "checkpoint"
    LOG_DIR = "logs"
    # 加载数据
    (X_train, y_train), (X_test, y_test) = load_data()

    # 数据处理
    X_train, y_train, X_test, y_test = process_data(X_train, y_train, X_test, y_test)

    # 创建模型
    model = LeNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)

    # 编译模型
    model.compile(loss='categorical_crossentropy',
                  optimizer=OPTIMIZER,
                  metrics=["accuracy"])

    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_PATH, "model-{epoch:02d}.h5")
    )

    tensorboard = TensorBoard(log_dir=LOG_DIR, histogram_freq=0, write_graph=True, write_images=True)

    history = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=NB_EPOCHS,
                        verbose=VERBOSE,
                        validation_split=VALIDATION_SPLIT,
                        callbacks=[checkpoint, tensorboard]
                        )
    model.save(MODEL_PATH + "/leNet_model.h5")
    score = model.evaluate(X_test, y_test, verbose=VERBOSE)

    print("Test score:", score[0])
    print("Test acc:", score[1])

    # 列出全部历史数据
    print(history.history.keys())
    # 汇总准确率历史数据
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(LOG_DIR+"/model_acc")
    plt.show()
    # 汇总损失函数历史数据
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(LOG_DIR+"/model_loss")
    plt.show()
