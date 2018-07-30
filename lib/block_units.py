from keras.models import Model
from keras.layers import *
from keras.layers.core import Activation, Layer

'''
Keras Customizable Residual Unit
This is a simplified implementation of the basic (no bottlenecks) full pre-activation residual unit from He, K., Zhang, X., Ren, S., Sun, J., "Identity Mappings in Deep Residual Networks" (http://arxiv.org/abs/1603.05027v2).
'''

def conv_block(feat_maps_out, prev):
    prev = BatchNormalization()(prev) # Specifying the axis and mode allows for later merging
    prev = Activation('relu')(prev)
    prev = Conv3D(feat_maps_out, (3, 3, 5), padding="same")(prev) 
    prev = BatchNormalization()(prev) # Specifying the axis and mode allows for later merging
    prev = Activation('relu')(prev)
    prev = Conv3D(feat_maps_out, (3, 3, 5), padding="same")(prev)   
 
    return prev


def identitiy_fix_size(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = Conv3D(feat_maps_out, (1, 1, 1), padding='same')(prev)
    return prev 


def Residual(feat_maps_in, feat_maps_out, prev_layer):
    '''
    A customizable residual unit with convolutional and shortcut blocks
    Args:
      feat_maps_in: number of channels/filters coming in, from input or previous layer
      feat_maps_out: how many output channels/filters this block will produce
      prev_layer: the previous layer
    '''

    id = identitiy_fix_size(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block(feat_maps_out, prev_layer)

    print('Residual block mapping '+str(feat_maps_in)+' channels to '+str(feat_maps_out)+' channels built')
    return merge([id, conv], mode='sum') # the residual connection


def dense_block(feat_maps_out, prev):
    prev = Dense(feat_maps_out, activation='relu', kernel_initializer='he_normal')(prev)
    prev = Dropout(rate=0.4)(prev)
    prev = BatchNormalization()(prev)
    prev = Dense(feat_maps_out, activation='relu', kernel_initializer='he_normal')(prev)
    prev = Dropout(rate=0.4)(prev)
    prev = BatchNormalization()(prev)
    return prev
    
def identitiy_fix_size_dense(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = Dense(feat_maps_out, activation='relu', kernel_initializer='he_normal')(prev)
    return prev


def Dense_Residual(feat_maps_in, feat_maps_out, prev_layer):
    '''
    A residual unit with dense blocks 
    Args:
      feat_maps_in: number of channels/filters coming in, from input or previous layer
      feat_maps_out: how many output channels/filters this block will produce
      prev_layer: the previous layer
    '''

    id = identitiy_fix_size_dense(feat_maps_in, feat_maps_out, prev_layer)
    dense = dense_block(feat_maps_out, prev_layer)

    return merge([id, dense], mode='sum') # the residual connection


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

def conv_3pyramide_wDrop_wBatchNorm(x0, n_kernels, drop=0.3, **kwargs):
    if len(n_kernels) != 3:
        print('Conv_3pyramide stacks three convolutions. Give array of 3\
              kernel-lengths!')
    x1 = Convolution3D(n_kernels[0], (3, 3, 5), padding='same', **kwargs)(x0)
    x1 = Dropout(rate=drop*0.6)(x1)
    x1 = BatchNormalization()(x1)
    x2 = Convolution3D(n_kernels[1], (2, 2, 3), padding='same', **kwargs)(x1)
    x2 = Dropout(rate=drop)(x2)
    x2 = BatchNormalization()(x2)
    x3 = Convolution3D(n_kernels[2], (2, 2, 2), padding='same', **kwargs)(x2)
    return x3

def conv_2pyramide_wDrop_wBatchNorm(x0, n_kernels, drop=0.3, **kwargs):
    if len(n_kernels) != 2:
        print('Conv_3pyramide stacks three convolutions. Give array of 3\
              kernel-lengths!')
    x1 = Convolution3D(n_kernels[0], (3, 3, 5), padding='same', **kwargs)(x0)
    x1 = Dropout(rate=drop*0.6)(x1)
    x1 = BatchNormalization()(x1)
    x2 = Convolution3D(n_kernels[1], (2, 2, 3), padding='same', **kwargs)(x1)
    x2 = Dropout(rate=drop)(x2)
    return x2

def conv_3pyramide_shortcutted(x0, n_kernels, **kwargs):
    if len(n_kernels) != 3:
        print('Conv_3pyramide stacks three convolutions. Give array of 3\
              kernel-lengths!')
    x1_a = Convolution3D(n_kernels[0], (3, 3, 5), padding='same', **kwargs)(x0)
    x1 = BatchNormalization()(x1_a)
    x2 = Convolution3D(n_kernels[1], (2, 2, 3), padding='same', **kwargs)(x1)
    x2 = Dropout(rate=drop)(x2)
    x2 = concatenate([x2, x1_a], axis=-1)
    x2 = BatchNormalization()(x2)
    x3 = Convolution3D(n_kernels[2], (2, 2, 2), padding='same', **kwargs)(x2)
    return x3



def inception_unit_pyramides(x0, **kwargs):

    x1 = conv_3pyramide(x0, [8, 16, 24], **kwargs)

    x2 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x0)
    x2 = conv_3pyramide(x0, [4, 6, 12], **kwargs)

    x3 = Convolution3D(16, (5, 5, 5), padding="same", **kwargs)(x0)
    x3 = conv_3pyramide(x3, [4, 6, 12], **kwargs)

    return concatenate([x1, x2, x3], axis=-1)

def triple_conv_block(x0, features=12,  kernels=(2,2,2), **kwargs):
    x1 = Convolution3D(features, kernels, padding='same', **kwargs)(x0)
    x1 = Convolution3D(features, kernels, padding='same', **kwargs)(x1)
    x1 = Convolution3D(features, kernels, padding='same', **kwargs)(x1)
    return x1

def triple_conv_block_wBN(x0, features=12,  kernels=(2,2,2), **kwargs):
    x1 = Convolution3D(features, kernels, padding='same', **kwargs)(x0)
    x1 = BatchNormalization()(x1)
    x1 = Convolution3D(features, kernels, padding='same', **kwargs)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Convolution3D(features, kernels, padding='same', **kwargs)(x1)
    return x1


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    Arguments:
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x
