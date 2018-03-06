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
sys.path.append("..")
import transformations as tr
import block_units
import numpy as np


# *Settings*
# define inputs for each branch
inputs = OrderedDict()

inputs["Branch1"] = {"variables": ["charge", "time"],
                     "transformations": [tr.centralize, tr.centralize]}
inputs["Branch2"] = {"variables": ["charge"],
                     "transformations": [np.sum]}
inputs["Branch3"] = {"variables": ["time"],
                     "transformations": [tr.max_min_delta]}


# define outputs for each branch
outputs = OrderedDict()
outputs["Out1"] = {"variables": ["classificationTag"],
                   "transformations": [tr.oneHotEncode_noDoubleBang]}

reference_outputs = []

# *Model*

def model(input_shapes, output_shapes):

    kwargs = dict(activation='elu', kernel_initializer='he_normal')

    #branch1
    input_b1 = Input(shape=input_shapes["Branch1"]["general"],
                     name = "Input-Branch1")
    z1 = Conv3D(12, (3, 3, 5), padding="same", **kwargs)(input_b1)
    z1 = MaxPooling3D(pool_size=(2, 2, 3))(z1)
    z1 = Conv3D(16, (2, 2, 4))(z1)
    z1 = Dropout(rate=0.2)(z1)
    z1 = BatchNormalization()(z1)
    z1 = Conv3D(24, (2, 2, 3))(z1)
    z1 = MaxPooling3D(pool_size=(1, 1, 2))(z1)
    z1 = Dropout(rate=0.2)(z1)
    z1 = Flatten()(z1)
    z1 = Dense(84, **kwargs)(z1)
    z1 = Dropout(rate=0.2)(z1)

    # branch 2
    input_b2 = Input(shape=input_shapes["Branch2"]["general"],
                     name="Input-Branch2")

    # branch 3 
    input_b3 = Input(shape=input_shapes["Branch3"]["general"],
                     name="Input-Branch3")




    # merge
    zo = concatenate([z1, input_b2, input_b3])
    zo = Dense(312,\
              **kwargs)(zo)
    zo = Dropout(rate=0.2)(zo)
    zo = BatchNormalization()(zo)
    zo = Dense(128, **kwargs)(zo)
    zo = Dropout(rate=0.2)(zo)
    zo = BatchNormalization()(zo)
    zo = Dense(64, **kwargs)(zo)


    # output
    output_layer1 = Dense(output_shapes["Out1"]["general"][0],\
                          activation="softmax",\
                          name="Target")(zo)
    model = keras.models.Model(inputs=[input_b1, input_b2, input_b3],\
                               outputs=[output_layer1])
    print(model.summary())
    return model
