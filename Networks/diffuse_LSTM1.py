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

inputs["Branch1"] = {"variables": ["charge"],
                     "transformations": [tr.identity]}

# define outputs for each branch

outputs = OrderedDict()
outputs["Out1"] = {"variables": ["energy"],#, "azimuth", "zenith"],
                   "transformations": [np.log10]}


reference_outputs = ['muex',\
                     'energy_splinempe_muon']

def model(input_shapes, output_shapes):

    kwargs = dict(activation='relu')
    dense_kwargs = dict(activation='relu')

    #branch1
    input_b1 = Input(shape=input_shapes["Branch1"]["general"],
                     name = "Input-Branch1")

    z1 = TimeDistributed(Conv3D(24, (3,3,4), **kwargs),
                        input_shape=input_shapes["Branch1"]["general"])(input_b1)
    conv1 = TimeDistributed(MaxPooling3D((1,1,3)))(z1)
    z1 = TimeDistributed(Conv3D(24, (3,3,3), padding="same", **kwargs))(conv1)
    z1 = TimeDistributed(MaxPooling3D((2,2,3)))(z1)
    z1 = TimeDistributed(Flatten())(z1)
    z1 = LSTM(32,
              activation='tanh', recurrent_activation='hard_sigmoid',
              # activation="softsign",
              use_bias=True, kernel_initializer='glorot_uniform',
              recurrent_initializer='orthogonal', bias_initializer='zeros',
              recurrent_dropout=0.0, implementation=1, return_sequences=True,
              )(z1)
    z1 = TimeDistributed(Dense(64, **kwargs))(z1)
    z1 = Dropout(rate=0.3)(z1)
    z1 = LSTM(32,
              activation='tanh', recurrent_activation='hard_sigmoid',
              # activation="softsign",
              use_bias=True, kernel_initializer='glorot_uniform',
              recurrent_initializer='orthogonal', bias_initializer='zeros',
              recurrent_dropout=0.0, implementation=1, return_sequences=False,
              )(z1)
    z1 = Dense(64, **kwargs)(z1)
    z1 = BatchNormalization()(z1)
    zo = Dense(32, **dense_kwargs)(z1)
    zo = Dropout(rate=0.32)(zo)
    # output
    output_layer1 = Dense(output_shapes["Out1"]["general"][0],\
                          activation="relu",\
                          name="Target")(zo)
    model = keras.models.Model(inputs=[input_b1],\
                               outputs=[output_layer1])
    print(model.summary())
    return model

