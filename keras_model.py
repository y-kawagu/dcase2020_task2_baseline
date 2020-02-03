########################################################################
# import python-library
########################################################################
# from import
import keras.models
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Activation

########################################################################
# keras model
########################################################################
def get_model(inputDim):
    """
    define the keras model
    the model based on the simple dense auto encoder (64*64*8*64*64)
    """
    inputLayer = Input(shape=(inputDim,))

    h = Dense(128)(inputLayer)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    
    h = Dense(8)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(128)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(inputDim, activation=None)(h)

    return Model(inputs=inputLayer, outputs=h)
#########################################################################


def load_model(file_path):
    return keras.models.load_model(file_path)

    