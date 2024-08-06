# < -----  Import Necessary Packages ----- >
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras import Input, Model

def CNN_Model(xtrain, ytrain):
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

    flat = Flatten()(pool3)
    dense_layer = Dense(128, 'relu')(flat)
    # output layer
    out = Dense(ytrain.shape[1], 'softmax')(den)

    # build model
    model = Model(inputs=Input_layer, outputs=out)
    model.summary()
    return model
    
