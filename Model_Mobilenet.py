from keras.models import Model
from keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, AveragePooling2D, Dense, Activation, Flatten, \
    Add, Dropout
from keras import backend as K
import numpy as np

from Evaluation import evaluation


def relu6(inputs):
    '''
        Performs the ReLU6 activation function for the bottleneck stage of the MobileNet V2
        Inputs:
            inputs: the layer with the inputs for the activation function
        Return:
            Min value between the value of the regular ReLU function and 6
    '''

    return K.relu(inputs, max_value=6)


def bottleneck(inputs, t, alpha, num_filters, kernel_sz=(3, 3), stride=(1, 1), pad='same', residual=False,
               dropout=False, dropout_perc=0.1):
    '''
        Performs the bottleneck stage of the MobileNet V2
        Inputs:
            inputs: the layer with the inputs
            t: the value used to increase the number of filters of the expansion stage
            alpha: width multiplier that controls the number of filters of the output tensor
            num_filters: number of filters of the output tensor
            kernel_sz = kernel size of the filter
            stride: stride of the kernel
            pad: padding of the filter
            residual: parameter that determine the sum of the input and output of the bottleneck stage
            dropout: determine if dropout will be performed
            dropout_perc: percentage of neurons that will be set to zero
        Return:
            x: the result of the bottleneck stage
    '''

    # Get the index of the input 4D tensor that represents the number of channels of the image
    # -1 can also represent the last element of the tensor
    channel_idx = 1 if K.image_data_format == 'channels_first' else -1

    # Number of filters for the expansion convolution
    num_filters_exp = K.int_shape(inputs)[channel_idx] * t

    # Number of filters of the projection convolution
    num_filters_proj = int(num_filters * alpha)

    # Expansion layer
    x = Conv2D(filters=num_filters_exp, kernel_size=(1, 1), strides=(1, 1), padding=pad)(inputs)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)

    # Depthwise convolution
    x = DepthwiseConv2D(kernel_size=kernel_sz, strides=stride, depth_multiplier=1, padding=pad)(x)
    x = BatchNormalization()(x)
    x = Activation(relu6)(x)

    # Projection convolution
    x = Conv2D(filters=num_filters_proj, kernel_size=(1, 1), strides=(1, 1), padding=pad)(x)
    x = BatchNormalization()(x)

    if (residual == True):
        x = Add()([x, inputs])

    if (dropout == True):
        x = Dropout(dropout_perc)(x)

    return x


def depthwise_block(inputs, stride, kernel_sz=(3, 3), pad='same'):
    '''
        Function that performs the depthwise convolution
        Inputs:
            inputs:    the input shape of the depthwise convolution
            kernel_sz: a tuple that indicates the size of the filtering kernel
            stride:    a tuple that indicates the strides of the kernel
        Return:
            x: the result of the depthwise convolution
    '''

    x = DepthwiseConv2D(kernel_size=kernel_sz, strides=stride, depth_multiplier=1, padding=pad)(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)

    return x


def pointwise_block(inputs, num_filters, alpha, kernel_sz=(1, 1), stride=(1, 1), pad='same', dropout=False,
                    dropout_perc=0.1):
    '''
        Function that performs the pointwise convolution
        Inputs:
            inputs:      the input shape of the depthwise convolution
            num_filters: number of filters to be used in the convolution
            kernel_sz:   a tuple that indicates the size of the filtering kernel
            stride:      a tuple that indicates the strides of the kernel
            dropout: determine if dropout will be performed
            dropout_perc: percentage of neurons that will be set to zero
        Return:
            x: the result of the pointwise convolution
    '''

    # Number of filters based on width multiplier reported in the original paper
    n_fil = int(num_filters * alpha)

    x = Conv2D(filters=n_fil, kernel_size=kernel_sz, padding=pad)(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)

    if (dropout == True):
        x = Dropout(dropout_perc)(x)

    return x


def MobileNet(Train_Data, Test_Target, input_shape, num_units, filters=32, kernel_sz=(3, 3), stride=(2, 2), alp=1, ro=1, dropout_perc=0.1):
    input_shape = (int(input_shape[0] * ro), int(input_shape[1] * ro), input_shape[2])

    inputs = Train_Data(shape=input_shape)

    # Regular convolution
    x = Conv2D(filters=filters, kernel_size=kernel_sz, strides=stride)(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = Dropout(dropout_perc)(x)

    # First depthwise-pointwise block
    x = depthwise_block(x, kernel_sz=(3, 3), stride=(1, 1))
    x = pointwise_block(x, num_filters=64, alpha=alp, stride=(1, 1), dropout=True, dropout_perc=dropout_perc)

    # Second depthwise-pointwise block
    x = depthwise_block(x, kernel_sz=(3, 3), stride=(2, 2))
    x = pointwise_block(x, num_filters=128, alpha=alp, stride=(1, 1), dropout=True, dropout_perc=dropout_perc)

    # Third depthwise-pointwise block
    x = depthwise_block(x, kernel_sz=(3, 3), stride=(1, 1))
    x = pointwise_block(x, num_filters=128, alpha=alp, stride=(1, 1), dropout=True, dropout_perc=dropout_perc)

    # Fourth depthwise-pointwise block
    x = depthwise_block(x, kernel_sz=(3, 3), stride=(2, 2))
    x = pointwise_block(x, num_filters=256, alpha=alp, stride=(1, 1), dropout=True, dropout_perc=dropout_perc)

    # Fifth depthwise-pointwise block
    x = depthwise_block(x, kernel_sz=(3, 3), stride=(1, 1))
    x = pointwise_block(x, num_filters=256, alpha=alp, stride=(1, 1), dropout=True, dropout_perc=dropout_perc)

    # Sixth depthwise-pointwise block
    x = depthwise_block(x, kernel_sz=(3, 3), stride=(2, 2))
    x = pointwise_block(x, num_filters=512, alpha=alp, stride=(1, 1), dropout=True, dropout_perc=dropout_perc)

    # Seventh depthwise-pointwise block (repeated five times)
    x = depthwise_block(x, kernel_sz=(3, 3), stride=(1, 1))
    x = pointwise_block(x, num_filters=512, alpha=alp, stride=(1, 1), dropout=True, dropout_perc=dropout_perc)

    x = depthwise_block(x, kernel_sz=(3, 3), stride=(1, 1))
    x = pointwise_block(x, num_filters=512, alpha=alp, stride=(1, 1), dropout=True, dropout_perc=dropout_perc)

    x = depthwise_block(x, kernel_sz=(3, 3), stride=(1, 1))
    x = pointwise_block(x, num_filters=512, alpha=alp, stride=(1, 1), dropout=True, dropout_perc=dropout_perc)

    x = depthwise_block(x, kernel_sz=(3, 3), stride=(1, 1))
    x = pointwise_block(x, num_filters=512, alpha=alp, stride=(1, 1), dropout=True, dropout_perc=dropout_perc)

    x = depthwise_block(x, kernel_sz=(3, 3), stride=(1, 1))
    x = pointwise_block(x, num_filters=512, alpha=alp, stride=(1, 1), dropout=True, dropout_perc=dropout_perc)

    # Eight depthwise-pointwise block
    x = depthwise_block(x, kernel_sz=(3, 3), stride=(2, 2))
    x = pointwise_block(x, num_filters=1024, alpha=alp, stride=(1, 1), dropout=True, dropout_perc=dropout_perc)

    # Nineth depthwise-pointwise block
    x = depthwise_block(x, kernel_sz=(3, 3), stride=(1, 1))
    x = pointwise_block(x, num_filters=1024, alpha=alp, stride=(1, 1), dropout=True, dropout_perc=dropout_perc)

    # Pooling layer
    # Pooling size correction due to the resolution multiplier parameter
    pool_size = int(np.round(7 * ro))
    x = AveragePooling2D(padding='valid', pool_size=(pool_size, pool_size), strides=(1, 1))(x)

    x = Flatten()(x)

    # Fully connected layer
    x = Dense(units=1024, activation='relu')(x)

    # Softmax layer
    output = Dense(num_units, activation='softmax')(x)

    model = Model(inputs, output)
    Eval = evaluation(output, Test_Target)
    pred = model.predict(Test_Target)

    return Eval, pred