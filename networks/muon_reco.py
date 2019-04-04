''' new Network definitions using the functional API from keras.
first part: model settings like input variables, outputs and transformations
second part: model definition, name must be def model(input_shape):
'''

import numpy as np
import keras
from keras.layers import *
from keras import regularizers
from keras.utils import to_categorical
import sys
from collections import OrderedDict
import os
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.join(os.path.abspath(".."),'lib'))
import transformations as tr
import numpy as np
import block_units as bunit


# *Settings*
# define inputs for each branch
inputs = OrderedDict()



inputs["Branch_IC_time"] = {"variables": ["IC_charge", "IC_first_charge", "IC_time_first", "IC_pulse_0_5_pct_charge_quantile"],
                     "transformations": [tr.identity, tr.identity, tr.centralize, tr.centralize]}

# define outputs for each branch
outputs = OrderedDict()
outputs["Out1"] = {"variables": ["mu_dir_x"],
                   "transformations": [tr.identity]}
outputs["Out2"] = {"variables": ["mu_dir_y"],
                   "transformations": [tr.identity]}
outputs["Out3"] = {"variables": ["mu_dir_z"],
                   "transformations": [tr.identity]}
outputs["Out4"] = {"variables": ["mu_e_on_entry"],
                   "transformations": [tr.log10]}
reference_outputs = []

# Step 3: Define loss functions

loss_weights = {'Target1': 1., 'Target2': 1, 'Target3':1, 'Target4':1}
loss_functions = ["mean_squared_error", "mean_squared_error", "mean_squared_error", "mean_squared_error"]


# Step 4: Define the model using Keras functional API
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
    x = bunit.conv3d_bn(input_tensor, 32, kernel, padding='same')
    x = bunit.conv3d_bn(x, 16, kernel, padding='same')
    x = bunit.conv3d_bn(x, 32, kernel)
    x = MaxPooling3D((1,1,2), strides=1)(x)
    x = bunit.conv3d_bn(x, 40, 1, padding='same')
    x = bunit.conv3d_bn(x, 48, kernel, padding='same')
    x = MaxPooling3D((2,2,3), strides=(1,1,3))(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = bunit.conv3d_bn(x, 48, 1, padding='same')
    branch_1 = bunit.conv3d_bn(x, 24, 1)
    branch_1 = bunit.conv3d_bn(branch_1, 24, 5, padding='same')
    branch_2 = bunit.conv3d_bn(x, 32, 1)
    branch_2 = bunit.conv3d_bn(branch_2, 48, 3)
    branch_2 = bunit.conv3d_bn(branch_2, 48, 3, padding='same')
    branch_pool = AveragePooling3D(3,1, padding='same')(x)
    branch_pool = bunit.conv3d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    print('imaga_data_format {}'.format(K.image_data_format()))
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 3):
        x = bunit.inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = bunit.conv3d_bn(x, 64, (2,2,3), strides=(2,2,3),padding='valid')
    branch_1 = bunit.conv3d_bn(x, 32, 1)
    branch_1 = bunit.conv3d_bn(branch_1, 32, 2)
    branch_1 = bunit.conv3d_bn(branch_1, 64, (2,2,3), strides=(2,2,3), padding='valid')
    branch_pool = MaxPooling3D((2,2,3),strides=(2,2,3) , padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 3):
        x = bunit.inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)

#    x = bunit.inception_resnet_block(x,
#                               scale=1.,
#                               activation=None,
#                               block_type='block8',
#                               block_idx=10)

    # Final convolution block: 8 x 8 x 1536
    x = bunit.conv3d_bn(x, 1800, 1, name='conv_7b')

    if pooling == 'avg':
        x = GlobalAveragePooling3D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling3D()(x)
    return x


def model(input_shapes, output_shapes):

    kwargs = dict(activation='relu', kernel_initializer='he_normal')


    #branch1
    input_b1 = Input(shape=input_shapes["Branch_IC_time"]["general"],
                     name = "Input-Branch1")
    z1 = InceptionResNetV2_small(input_b1, pooling='avg')
    output_b1 = Dense(output_shapes["Out1"]["general"][0],\
                          activation="relu",\
                          name="Target1")(z1)
    output_b2 = Dense(output_shapes["Out2"]["general"][0],\
                          activation="relu",\
                          name="Target2")(z1)
    output_b3 = Dense(output_shapes["Out3"]["general"][0],\
                          activation="relu",\
                          name="Target3")(z1)
    output_b4 = Dense(output_shapes["Out4"]["general"][0],\
                          activation="relu",\
                          name="Target4")(z1)


    model = keras.models.Model(inputs=[input_b1],
                               outputs=[output_b1, output_2,
                                        output_3, output_4])
    print(model.summary())
    return model

