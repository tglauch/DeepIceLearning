''' new Network definitions using the functional API from keras.
first part: model settings like input variables, outputs and transformations
second part: model definition, name must be def model(input_shape):
'''

import numpy as np
import keras
from keras.layers import *
import sys
from collections import OrderedDict
sys.path.append("..")
import transformations as tr
import special_layers

# *Settings*
# define inputs for each branch

inputs = OrderedDict()

inputs["Branch1"] = {"variables": ["charge", "time", "time_spread"],
                     "transformations": [tr.identity, tr.identity , tr.identity]}
#tr.centralize]}
inputs["Branch2"] = {"variables": ["charge"],
                     "transformations": [np.sum]}

# define outputs for each branch

outputs = OrderedDict()
outputs["Out1"] = {"variables": ["energy"],#, "azimuth", "zenith"],
                   "transformations": [np.log10]}
#, tr.identity, tr.identity]}
# outputs["Out2"] = {"variables": ["azimuth"],
#                    "transformations": [np.log10]}


# *Model*

def model(input_shapes, output_shapes):

    kwargs = dict(activation='relu', kernel_initializer='he_normal')

    # branch 1
    input_b1 = Input(shape=input_shapes["Branch1"]["general"],
                     name="Input-Branch1")

    z = special_layers.inception_unit_pyramides(input_b1, **kwargs)
    z = BatchNormalization()(z)
    z = MaxPooling3D(pool_size=(2, 1, 3))(z)
    z = Conv3D(24, (2, 2, 2), **kwargs)(z)
    z = Flatten()(z)

    # branch 2
    input_b2 = Input(shape=input_shapes["Branch2"]["general"],
                     name="Input-Branch2")

    # merge
    z = concatenate([z, input_b2])

    z = Dense(256, **kwargs)(z)
    z = Dropout(rate=0.2)(z)
    z = BatchNormalization()(z)
    z = Dense(128, **kwargs)(z)
    z = Dropout(rate=0.2)(z)
    z = BatchNormalization()(z)
    z = Dense(64, **kwargs)(z)

    # output
    output_layer1 = Dense(output_shapes["Out1"]["general"],\
                          activation="linear",\
                          name="Target")(z)

    model = keras.models.Model(
        inputs=[input_b1, input_b2],
        outputs=[output_layer1])
    print(model.summary())
    return model
