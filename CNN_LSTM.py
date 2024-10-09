# < -----  Import Necessary Packages ----- >
from keras.layers import *
from keras import Input, Model
import numpy as np



def CNN_LSTM(xtrain, ytrain):
    """
        Args:
            xtrain: image array with 4 dimension input
            ytrain: labels of corresponding images with 2 dimension input

        Returns:
            keras functional model
        """
    # functional model
    # Input layer
    Input_layer = Input(shape=(xtrain.shape[1], xtrain.shape[2], xtrain.shape[3]))

    # series of convolution and maxpooling layer
    conv1 = Conv2D(16, (3, 3), activation='relu')(Input_layer)
    pool1 = MaxPooling2D(2, 2)(conv1)
    conv2 = Conv2D(32, (3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D(2, 2)(conv2)
    conv3 = Conv2D(64, (3, 3), activation='relu')(pool2)
    pool3 = MaxPooling2D(2, 2)(conv3)

    flat = TimeDistributed(Flatten())(pool3)

    LSTM_layer1 = LSTM(10, return_sequences=True)(flat)
    LSTM_layer2 = LSTM(20, return_sequences=False)(LSTM_layer1)

    dense_layer = Dense(128, 'relu')(LSTM_layer2)
    # output layer
    out = Dense(ytrain.shape[1], 'softmax')(dense_layer)

    # build model
    model = Model(inputs=Input_layer, outputs=out)
    model.summary()
    return model
