from keras.models import Model
from keras.layers import *
from keras.layers.core import Activation, Layer


def conv_block(feat_maps_out, prev, kernel_size=(3, 3, 5)):
    # Specifying the axis and mode allows for later merging
    prev = BatchNormalization()(prev)
    prev = Activation('relu')(prev)
    prev = Conv3D(feat_maps_out,
                  kernel_size,
                  padding="same")(prev)
    prev = BatchNormalization()(prev)
    prev = Activation('relu')(prev)
    prev = Conv3D(feat_maps_out,
                  kernel_size,
                  padding="same")(prev)

    return prev


def identitiy_fix_size(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution
        # on shortcuts that map between an uneven amount of channels
        prev = Conv3D(feat_maps_out, (1, 1, 1), padding='same')(prev)
    return prev


def Residual(feat_maps_in, feat_maps_out, prev_layer, kernel_size=(3, 3, 5)):
    '''
    A customizable residual unit with convolutional and shortcut blocks
    Args:
      feat_maps_in: number of channels/filters coming in, from input or previous layer
      feat_maps_out: how many output channels/filters this block will produce
      prev_layer: the previous layer
    '''

    id = identitiy_fix_size(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block(feat_maps_out, prev_layer, kernel_size)

    print('Residual block mapping '+str(feat_maps_in)+' channels to '+str(feat_maps_out)+' channels built')
    return concatenate([id, conv]) # the residual connection


def dense_block(feat_maps_out, prev):
    prev = Dense(feat_maps_out,
                 activation='relu',
                 kernel_initializer='he_normal')(prev)
    prev = Dropout(rate=0.4)(prev)
    prev = BatchNormalization()(prev)
    prev = Dense(feat_maps_out,
                 activation='relu',
                 kernel_initializer='he_normal')(prev)
    prev = Dropout(rate=0.4)(prev)
    prev = BatchNormalization()(prev)
    return prev


def identitiy_fix_size_dense(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = Dense(feat_maps_out,
                     activation='relu',
                     kernel_initializer='he_normal')(prev)
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

    return add([id, dense])  # the residual connection


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

    return add([x1, x2, x4], axis=-1)


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
    x1 = Dropout(rate=drop * 0.6)(x1)
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
    x1 = Dropout(rate=drop * 0.6)(x1)
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


def triple_conv_block(x0, features=12, kernels=(2, 2, 2), **kwargs):
    x1 = Convolution3D(features, kernels, padding='same', **kwargs)(x0)
    x1 = Convolution3D(features, kernels, padding='same', **kwargs)(x1)
    x1 = Convolution3D(features, kernels, padding='same', **kwargs)(x1)
    return x1


def triple_conv_block_wBN(x0, features=12, kernels=(2, 2, 2), **kwargs):
    x1 = Convolution3D(features, kernels, padding='same', **kwargs)(x0)
    x1 = BatchNormalization()(x1)
    x1 = Convolution3D(features, kernels, padding='same', **kwargs)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Convolution3D(features, kernels, padding='same', **kwargs)(x1)
    return x1


def conv3d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = Conv3D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    """Adds a Inception-ResNet block.
    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='block35'`
        - Inception-ResNet-B: `block_type='block17'`
        - Inception-ResNet-C: `block_type='block8'`
    # Arguments
        x: input tensor.
        scale: scaling factor to scale the residuals (i.e., the output of
            passing `x` through an inception module) before adding them
            to the shortcut branch. Let `r` be the output from the residual branch,
            the output of this block will be `x + scale * r`.
        block_type: `'block35'`, `'block17'` or `'block8'`, determines
            the network structure in the residual branch.
        block_idx: an `int` used for generating layer names. The Inception-ResNet blocks
            are repeated many times in this network. We use `block_idx` to identify
            each of the repetitions. For example, the first Inception-ResNet-A block
            will have `block_type='block35', block_idx=0`, ane the layer names will have
            a common prefix `'block35_0'`.
        activation: activation function to use at the end of the block
            (see [activations](keras./activations.md)).
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).
    # Returns
        Output tensor for the block.
    # Raises
        ValueError: if `block_type` is not one of `'block35'`,
            `'block17'` or `'block8'`.
    """
    if block_type == 'block35':
        branch_0 = conv3d_bn(x, 32, 1)
        branch_1 = conv3d_bn(x, 32, 1)
        branch_1 = conv3d_bn(branch_1, 32, 3)
        branch_2 = conv3d_bn(x, 32, 1)
        branch_2 = conv3d_bn(branch_2, 48, 3)
        branch_2 = conv3d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv3d_bn(x, 192, 1)
        branch_1 = conv3d_bn(x, 128, 1)
        branch_1 = conv3d_bn(branch_1, 160, (1, 1, 3))
        branch_1 = conv3d_bn(branch_1, 192, (3, 3, 1))
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv3d_bn(x, 192, 1)
        branch_1 = conv3d_bn(x, 192, 1)
        branch_1 = conv3d_bn(branch_1, 224, (1, 1 ,3))
        branch_1 = conv3d_bn(branch_1, 256, (2, 2, 1))
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    mixed = Concatenate(axis=channel_axis,
                        name=block_name + '_mixed')(branches)
    up = conv3d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=block_name + '_conv')

    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale},
               name=block_name)([x, up])
    if activation is not None:
        x = Activation(activation, name=block_name + '_ac')(x)
    return x


def InceptionResNetV2_small(input_tensor=None,
                      pooling=None,
		      kernel = (2,2,3)):
    """Instantiates the Inception-ResNet v2 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that when using TensorFlow, for best performance you should
    set `"image_data_format": "channels_last"` in your Keras config
    at `~/.keras/keras.json`.
    The model and the weights are compatible with both TensorFlow and Theano
    backends (but not CNTK). The data format convention used by the model is
    the one specified in your Keras config file.
    Note that the default input image size for this model is 299x299, instead
    of 224x224 as in the VGG16 and ResNet models. Also, the input preprocessing
    function is different (i.e., do not use `imagenet_utils.preprocess_input()`
    with this model. Use `preprocess_input()` defined in this module instead).
    # Arguments
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the last convolutional layer.
            - `'avg'` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `'max'` means that global max pooling will be applied.
    # Returns
        Output tensor for the block
    """

    # Stem block: 35 x 35 x 192
    x = conv3d_bn(input_tensor, 32, kernel, padding='same')
    x = conv3d_bn(x, 32, kernel, padding='same')
    x = conv3d_bn(x, 64, kernel)
    x = MaxPooling3D((1,1,2), strides=1)(x)
    x = conv3d_bn(x, 80, 1, padding='same')
    x = conv3d_bn(x, 96, kernel, padding='same')
    x = MaxPooling3D((2,2,3), strides=(1,1,2))(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv3d_bn(x, 96, 1, padding='same')
    branch_1 = conv3d_bn(x, 48, 1)
    branch_1 = conv3d_bn(branch_1, 64, 5, padding='same')
    branch_2 = conv3d_bn(x, 64, 1)
    branch_2 = conv3d_bn(branch_2, 96, 3)
    branch_2 = conv3d_bn(branch_2, 96, 3, padding='same')
    branch_pool = AveragePooling3D(3,1, padding='same')(x)
    branch_pool = conv3d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    print('imaga_data_format {}'.format(K.image_data_format()))
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 4):
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv3d_bn(x, 192, (2,2,3), strides=(2,2,2),padding='valid')
    branch_1 = conv3d_bn(x, 128, 1)
    branch_1 = conv3d_bn(branch_1, 128, 2)
    branch_1 = conv3d_bn(branch_1, 192, (2,2,3), strides=(2,2,2), padding='valid')
    branch_pool = MaxPooling3D((2,2,3),strides=(2,2,2) , padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 6):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)

    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10)

    # Final convolution block: 8 x 8 x 1536
    x = conv3d_bn(x, 1024, 1, name='conv_7b')

    if pooling == 'avg':
        x = GlobalAveragePooling3D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling3D()(x)
    return x
