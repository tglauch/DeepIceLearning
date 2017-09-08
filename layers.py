import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D,\
    BatchNormalization, MaxPooling2D, Convolution3D, MaxPooling3D, Merge
from keras import regularizers

def residual_unit(x0):

    x = Convolution3D(64, (1, 1, 1), border_mode="same", activation="relu")(x0)
    x = Convolution3D(64, (3, 3, 3), border_mode="same", activation="relu")(x0)
    return Merge.add([x, x0])


def dense_block(x0, n=8):
    """ Create a block of n densely connected pairs of convolutions """
    for i in range(n):
        x = Convolution3D(8, (3, 3, 3), padding='same', activation='relu')(x0)
        x0 = Merge.concatenate([x0, x], axis=-1)
    return x


def inception_unit(x0):
    x1 = Convolution3D(16, (1, 1, 1), padding='same', activation='relu')(x0)
    x1 = Convolution3D(16, (3, 3, 3), padding='same', activation='relu')(x1)

    x2 = Convolution3D(16, (1, 1, 1), padding='same', activation='relu')(x0)
    x2 = Convolution3D(16, (5, 5, 5), padding='same', activation='relu')(x2)

    x3 = Convolution3D(16, (1, 1, 1), padding='same', activation='relu')(x0)

    x4 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x0)
    x4 = Convolution3D(64, (1, 1, 1), padding='same', activation='relu')(x4)


    return Merge.concatenate([x1, x2, x3, x4], axis=1)

