import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D,\
    BatchNormalization, MaxPooling2D, Convolution3D, MaxPooling3D, merge
from keras import regularizers

def residual_unit(x0):

    x = Convolution3D(64, (1, 1, 1), border_mode="same", activation="relu")(x0)
    x = Convolution3D(64, (3, 3, 3), border_mode="same", activation="relu")(x0)
    return merge.Add([x, x0])


def dense_block(x0, n=8):
    """ Create a block of n densely connected pairs of convolutions """
    for i in range(n):
        x = Convolution3D(8, (3, 3, 3), padding='same', activation='relu')(x0)
        x0 = merge([x0, x], mode="concat", concat_axis=-1)
    return x

def inception_unit(x0):
    x1 = Convolution3D(16, (1, 1, 1), padding='same', activation='relu')(x0)
    x1 = Convolution3D(16, (3, 3, 3), padding='same', activation='relu')(x1)

    x2 = Convolution3D(16, (1, 1, 1), padding='same', activation='relu')(x0)
    x2 = Convolution3D(16, (5, 5, 5), padding='same', activation='relu')(x2)

    x3 = Convolution3D(16, (1, 1, 1), padding='same', activation='relu')(x0)

    x4 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x0)
    x4 = Convolution3D(64, (1, 1, 1), padding='same', activation='relu')(x4)

    return merge([x1, x2, x3, x4], mode="concat", concat_axis=-1)

def conv_3pyramide(x0, n_kernels, **kwargs):
    if len(n_kernels)!=3:
        print('Conv_3pyramide stacks three convolutions. Give array of 3\
              kernel-lengths!')
    x1 = Convolution3D(n_kernels[0], (3,3,4), padding='same', **kwargs)(x0)
    x2 = Convolution3D(n_kernels[1], (2,2,3), padding='same', **kwargs)(x1)
    x3 = Convolution3D(n_kernels[2], (2,2,2), padding='same', **kwargs)(x2)
    return x3

def inception_unit_pyramides(x0, **kwargs):

    x1 = conv_3pyramide(x0, [8,16,24], **kwargs)

    x2 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x0)
    x2 = conv_3pyramide(x0, [6,12,18], **kwargs)

    x3 = Convolution3D(16, (5, 5, 5), padding="same", **kwargs)(x0)
    x3 = conv_3pyramide(x3, [6,12,18], **kwargs)
    return merge([x1, x2, x3], mode="concat", concat_axis=-1)

