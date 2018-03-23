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
sys.path.append("/scratch9/mkron/software/DeepIceLearning/model_additions")
import residual_unit as resid


# *Settings*
# define inputs for each branch
inputs = OrderedDict()

inputs["Branch1"] = {"variables": ["charge", "first_charge", "time"],
                     "transformations": [tr.centralize, tr.centralize, tr.centralize]}
inputs["Branch2"] = {"variables": ["charge"],
                     "transformations": [tr.log_of_sum]}

inputs["Branch3"] = {"variables": ["time", "time_spread", "av_time_charges", "charge"],
                     "transformations": [tr.centralize, tr.centralize, tr.centralize, tr.centralize]}
inputs["Branch4"] = {"variables": ["time"],
                     "transformations": [tr.max_min_delta_log]}


# define outputs for each branch
outputs = OrderedDict()
outputs["Out1"] = {"variables": ["classificationTag"],
                   "transformations": [tr.oneHotEncode_noDoubleBang]}

reference_outputs = []

# *Model*

def model(input_shapes, output_shapes):

    kwargs = dict(activation='relu', kernel_initializer='he_normal')

    #branch1
    input_b1 = Input(shape=input_shapes["Branch1"]["general"],
                     name = "Input-Branch1")
    z1 = Conv3D(72, (3, 3, 5), padding="same", **kwargs)(input_b1)
    z1 = MaxPooling3D(pool_size=(2, 2, 3))(z1)
    z1 = BatchNormalization()(z1)
    z1 = resid.Residual(72, 72, z1)
    z1 = BatchNormalization()(z1)
    z1 = resid.Residual(72, 32, z1)
    z1 = BatchNormalization()(z1)
    z1 = resid.Residual(32, 32, z1)
    z1 = BatchNormalization()(z1)
    z1 = resid.Residual(32, 32, z1)
    z1 = BatchNormalization()(z1)
    z1 = Flatten()(z1)
    z1 = Dense(64, **kwargs)(z1)
    z1 = Dropout(rate=0.4)(z1)


    # branch 2
    input_b2 = Input(shape=input_shapes["Branch2"]["general"],
                     name="Input-Branch2")

    # merge #1 of Branche "charge"
    merge1 = concatenate([z1, input_b2])
    merge1 = Dense(36,\
              **kwargs)(merge1)
    merge1 = Dropout(rate=0.4)(merge1)
    merge1 = BatchNormalization()(merge1)
    


    #branch3
    input_b3 = Input(shape=input_shapes["Branch3"]["general"],
                     name = "Input-Branch3")
    z3 = Conv3D(72, (3, 3, 5), padding="same", **kwargs)(input_b3)
    z3 = MaxPooling3D(pool_size=(2, 2, 3))(z3)
    z3 = BatchNormalization()(z3)
    z3 = resid.Residual(72, 72, z3)
    z3 = BatchNormalization()(z3)
    z3 = resid.Residual(72, 32, z3)
    z3 = BatchNormalization()(z3)
    z3 = resid.Residual(32, 32, z3)
    z3 = BatchNormalization()(z3)
    z3 = resid.Residual(32, 32, z3)
    z3 = BatchNormalization()(z3)
    z3 = Flatten()(z3)
    z3 = Dense(64, **kwargs)(z3)
    z3 = Dropout(rate=0.4)(z3)

    # branch 4 
    input_b4 = Input(shape=input_shapes["Branch4"]["general"],
                     name="Input-Branch4")

    # merge #2 of Branche "time"
    merge2 = concatenate([z3, input_b4])
    merge2 = Dense(72,\
              **kwargs)(merge2)
    merge2 = Dropout(rate=0.4)(merge2)
    merge2 = BatchNormalization()(merge2)


    # merge total
    zo = concatenate([merge1, merge2])
    zo = resid.Dense_Residual(108, 72, zo)  # #input shape has to match the previous layer
    zo = resid.Dense_Residual(72, 72, zo)
    zo = resid.Dense_Residual(72, 72, zo)
    zo = resid.Dense_Residual(72, 36, zo)


    # output
    output_layer1 = Dense(output_shapes["Out1"]["general"][0],\
                          activation="softmax",\
                          name="Target")(zo)
    model = keras.models.Model(inputs=[input_b1, input_b2, input_b3, input_b4],\
                               outputs=[output_layer1])
    print(model.summary())
    return model
