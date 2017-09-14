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

# *Settings*
# define inputs for each branch

inputs = OrderedDict()

inputs["Branch1"] = {"variables": ["charge", "time"],
                     "transformations": [tr.identity, tr.identity]}
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

reference_outputs = ['muex']

# *Model*

def model(input_shapes, output_shapes):

    kwargs = dict(activation='relu', kernel_initializer='he_normal')

    input_b1 = Input(shape=input_shapes["Branch1"]["general"],

                     name="Input-Branch1")

    z = Conv3D(8, (2, 2, 3), **kwargs)(input_b1)

    z = MaxPooling3D(pool_size=(3, 3, 3))(z)

    z = Flatten()(z)

    input_b2 = Input(shape=input_shapes["Branch2"]["general"],
                     name="Input-Branch2")

    z = concatenate([z, input_b2])
    z = Dense(512)(z)
    output_layer1 = Dense(output_shapes["Out1"]["general"][0], name="Target")(z)

    model = keras.models.Model(
        inputs=[input_b1, input_b2],
        outputs=[output_layer1])
    print(model.summary())
    return model
