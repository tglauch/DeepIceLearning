''' new Network definitions using the functional API from keras.
first part: model settings like input variables, outputs and transformations
second part: model definition, name must be def model(input_shape):
'''

import numpy as np
import keras
from keras.layers import *
from keras import regularizers
import sys
from collections import OrderedDict
sys.path.append("..")
import transformations as tr
import block_units
import numpy as np

# *Settings*
# define inputs for each branch

inputs = OrderedDict()

inputs["Branch1"] = {"variables": ["charge", "time"],#, "time_spread"],
                     "transformations": [tr.identity, tr.identity]}# , tr.identity]}
#tr.centralize]}
inputs["Branch2"] = {"variables": ["charge"],
                     "transformations": [np.sum]}

inputs["Branch3"] = {"variables": ["charge"],
                     "transformations": [tr.sort_input]}


# define outputs for each branch

outputs = OrderedDict()
outputs["Out1"] = {"variables": ["energy"],#, "azimuth", "zenith"],
                   "transformations": [np.log10]}
#, tr.identity, tr.identity]}
# outputs["Out2"] = {"variables": ["azimuth"],
#                    "transformations": [np.log10]}

def crop(dim, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    def func(x):
        if dim==0:
            return x[start:end]
        if dim==1:
            return x[:, start:end]
        if dim==2:
            return x[:,:, start:end]
        if dim==3:
            return x[:,:,:,start:end]
    return Lambda(func)

# *Model*

def model(input_shapes, output_shapes):

    kwargs = dict(activation='relu', kernel_initializer='he_normal')

    # branch 1
    input_b1 = Input(shape=input_shapes["Branch1"]["general"],
                     name="Input-Branch1")

    z = block_units.inception_unit_pyramides(input_b1, **kwargs)
    z = BatchNormalization()(z)
    z = MaxPooling3D(pool_size=(2, 2, 4))(z)
    z = noise.GaussianNoise(0.05)(z)
    z = Conv3D(24, (2, 2, 2), **kwargs)(z)
    #z = MaxPooling3D(pool_size=(2,2,2))(z)
    z = Flatten()(z)

    # branch 2
    input_b2 = Input(shape=input_shapes["Branch2"]["general"],
                     name="Input-Branch2")

    # branch 3
    input_b3 = Input(shape=input_shapes["Branch3"]["general"],
                     name="Input-Branch3")
    z3 = input_b3 #Dense(24, **kwargs)(input_b3)

    # merge
    z = concatenate([z, input_b2, z3])

    z = Dense(256,\
              **kwargs)(z)
    z = Dropout(rate=0.3)(z)
    z = BatchNormalization()(z)
    z = Dense(128,\
              activity_regularizer=regularizers.l1(0.04),\
              **kwargs)(z)
    z = Dropout(rate=0.2)(z)
    z = BatchNormalization()(z)
    z = Dense(64, **kwargs)(z)

    # output
    output_layer1 = Dense(output_shapes["Out1"]["general"],\
                          activation="linear",\
                          name="Target")(z)

    model = keras.models.Model(
        inputs=[input_b1, input_b2, input_b3],
        outputs=[output_layer1])
    print(model.summary())
    return model
