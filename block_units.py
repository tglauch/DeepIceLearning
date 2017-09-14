#!/usr/bin/env python
# coding: utf-8

from keras.layers import *
from keras.layers.merge import *


def residual_unit(x0):

    x = Convolution3D(64, (1, 1, 1), border_mode="same", activation="relu")(x0)
    x = Convolution3D(64, (3, 3, 3), border_mode="same", activation="relu")(x0)
    return add([x, x0])


def dense_block(x0, n=8):
    """ Create a block of n densely connected pairs of convolutions """
    for i in range(n):
        x = Convolution3D(8, (3, 3, 3), padding='same', activation='relu')(x0)
        x0 = concatenade([x0, x], axis=-1)
    return x


def inception_unit(nfilters, x0, strides=(1, 1, 1)):
    x1 = Convolution3D(
        nfilters, (1, 1, 1), padding='same', activation='relu')(x0)
    x1 = Convolution3D(
        nfilters, (3, 3, 3), strides=strides,
        padding='same', activation='relu')(x1)

    x2 = Convolution3D(
        nfilters, (1, 1, 1), padding='same',
        activation='relu')(x0)
    x2 = Convolution3D(
        nfilters, (5, 5, 5), strides=strides,
        padding='same', activation='relu')(x2)

    # x3 = Convolution3D(
    #     nfilters, (1, 1, 1), padding='same', activation='relu')(x0)
    # x3 = Convolution3D(
    #     nfilters, (7, 7, 7), strides=strides,
    #     padding='same', activation='relu')(x0)

    x4 = MaxPooling3D((3, 3, 3), strides=strides, padding='same')(x0)
    x4 = Convolution3D(
        nfilters, (1, 1, 1), padding='same', activation='relu')(x4)

    return concatenate([x1, x2, x4], axis=-1)


def conv_3pyramide(x0, n_kernels, **kwargs):
    if len(n_kernels) != 3:
        print('Conv_3pyramide stacks three convolutions. Give array of 3\
              kernel-lengths!')
    x1 = Convolution3D(n_kernels[0], (3, 3, 5), padding='same', **kwargs)(x0)
    x2 = Convolution3D(n_kernels[1], (2, 2, 3), padding='same', **kwargs)(x1)
    x3 = Convolution3D(n_kernels[2], (2, 2, 2), padding='same', **kwargs)(x2)
    return x3


def inception_unit_pyramides(x0, **kwargs):

    x1 = conv_3pyramide(x0, [8, 16, 24], **kwargs)

    x2 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x0)
    x2 = conv_3pyramide(x0, [4, 6, 12], **kwargs)

    x3 = Convolution3D(16, (5, 5, 5), padding="same", **kwargs)(x0)
    x3 = conv_3pyramide(x3, [4, 6, 12], **kwargs)

    return concatenate([x1, x2, x3], axis=-1)

