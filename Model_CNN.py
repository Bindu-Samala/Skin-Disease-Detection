import cv2 as cv
import numpy as np
import tflearn
from tensorflow.python.framework import ops
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from Evaluation import evaluation
from keras import activations
import tensorflow as tf


def Model_CNN(train_data, train_target, test_data, test_target, sol=None):
    if sol is None:
        sol = [128, 5, 0.01]
    IMG_SIZE = 20
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 1))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 1))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))
    pred, activation = Model(Train_X, train_target, Test_X, test_target, sol)

    pred = np.asarray(pred)
    Eval = evaluation(pred, test_target)
    feat = np.asarray(activation)
    feat = np.reshape(feat, (feat.shape[0] * feat.shape[1], feat.shape[2] * feat.shape[3]))
    feat = np.resize(feat, (train_data.shape[0] + test_data.shape[0], 1000))

    return Eval, feat


def Model(X, Y, test_x, test_y, sol):
    LR = 1e-3
    ops.reset_default_graph()
    convnet = input_data(shape=[None, 20, 20, 1], name='input')

    convnet = conv_2d(convnet, 32, 5, name='layer-conv1', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, name='layer-conv2', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 5, 5, name='layer-conv3', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnetc = conv_2d(convnet, 64, 5, name='layer-conv4', activation='linear')
    convnet = max_pool_2d(convnetc, 5)

    convnet = conv_2d(convnet, 32, 5, name='layer-conv5', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet1 = fully_connected(convnet, 1024, name='layer-conv', activation='linear')
    convnet2 = dropout(convnet1, 0.8)

    convnet3 = fully_connected(convnet2, Y.shape[1], name='layer-conv-before-softmax', activation='linear')

    regress = regression(convnet3, optimizer='sgd', learning_rate=sol[2],
                         loss='mean_square', name='target')

    model = tflearn.DNN(regress, tensorboard_dir='log')

    MODEL_NAME = 'test.model'.format(LR, '6conv-basic')
    model.fit({'input': X}, {'target': Y}, n_epoch=5,
              validation_set=({'input': test_x}, {'target': test_y}),
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    pred = model.predict(test_x)
    activation = activations.relu(convnetc.W).numpy()
    activation = activation.eval(session=tf.compat.v1.Session())
    return pred
