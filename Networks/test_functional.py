''' new Network definitions using the functional API from keras.
first part: model settings like input variables, outputs and transformations
second part: model definition, name must be def model(input_shape):
'''

import numpy as np
import keras
from keras.layers import *
import sys
sys.path.append("..")
import transformations as tr
# transformations import identity, centralize

# *Settings*

# define inputs for each branch
inputs = dict()
inputs["Branch1"] = {"variables": ["charge", "time"],
                     "transformations": [tr.identity, tr.centralize]}
inputs["Branch2"] = {"variables": ["charge"],
                     "transformations": [np.sum]}
# define outputs for each branch
outputs = dict()
outputs["Out1"] = {"variables": ["energy"],
                   "transformations": [np.log10]}
outputs["Out2"] = {"variables": ["azimuth"],
                   "transformations": [np.log10]}

# *Model*

def model(input_shapes, output_shapes):

    kwargs = dict(activation='relu', kernel_initializer='he_normal')
    input_b1 = Input(shape=input_shapes["Branch1"]["general"],
                     name="Input-Branch1")

    z = Conv3D(8, (2, 2, 3), **kwargs)(input_b1)

    z = MaxPooling3D(pool_size=(3, 3, 3))(z)

    z = Flatten()(z)
    z = Dense(512)(z)
    output_layer = Dense(output_shapes["Out1"]["general"], name="Target")(z)

    model = keras.models.Model(inputs=[input_b1], outputs=[output_layer])
    print(model.summary())
    return model
