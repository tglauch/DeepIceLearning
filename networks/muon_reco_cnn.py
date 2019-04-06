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



inputs["Branch_IC_time"] = {"variables": ["IC_charge", "IC_first_charge", "IC_time_first",
                                          "IC_pulse_0_5_pct_charge_quantile", "IC_num_pulses"],
                     "transformations": [tr.identity, tr.identity, tr.centralize, tr.centralize, tr.identity]}

# define outputs for each branch
outputs = OrderedDict()
#outputs["Out1"] = {"variables": ["mu_dir_x"],
#                   "transformations": [tr.identity]}
#outputs["Out2"] = {"variables": ["mu_dir_y"],
#                   "transformations": [tr.identity]}
#outputs["Out3"] = {"variables": ["mu_dir_z"],
#                   "transformations": [tr.identity]}
outputs["Out1"] = {"variables": ["mu_e_on_entry"],
                   "transformations": [tr.log10]}
reference_outputs = []

# Step 3: Define loss functions

loss_weights = {'Target1': 1.} #, 'Target2': 1, 'Target3':1, 'Target4':1}
loss_functions = ["mean_squared_error"] #, "mean_squared_error", "mean_squared_error", "mean_squared_error"]


# Step 4: Define the model using Keras functional API
def cnn_model(input_tensor, **kwargs):

    z1 = Conv3D(72, (3, 3, 5), padding="same", **kwargs)(input_tensor)
    z1 = MaxPooling3D(pool_size=(2, 2, 3))(z1)
    z1 = BatchNormalization()(z1)
    z1 = bunit.Residual(32, 32, z1)
    z1 = BatchNormalization()(z1)
    z1 = Dropout(rate=0.4)(z1)
    z1 = bunit.Residual(32, 32, z1)
    z1 = BatchNormalization()(z1)
    z1 = MaxPooling3D(pool_size=(1, 1, 2))(z1)
    z1 = bunit.Residual(32, 16, z1)
    z1 = BatchNormalization()(z1)
    z1 = bunit.Residual(16, 16, z1)
    z1 = BatchNormalization()(z1)
    z1 = Dropout(rate=0.4)(z1)
    z1 = bunit.Residual(16, 16, z1)
    z1 = BatchNormalization()(z1)
    z1 = Flatten()(z1)
    z1 = Dense(16, **kwargs)(z1)
    z1 = Dropout(rate=0.4)(z1)
    o1 = bunit.Dense_Residual(16, 16, z1)
    o1 = bunit.Dense_Residual(16, 16, o1)
    o1 = bunit.Dense_Residual(16, 16, o1)
    o1 = Dropout(rate=0.4)(o1)
    o1 = bunit.Dense_Residual(16, 8, o1)
    o1 = bunit.Dense_Residual(8, 8, o1)
    return o1


def model(input_shapes, output_shapes):

    kwargs = dict(activation='relu', kernel_initializer='he_normal')


    #branch1
    input_b1 = Input(shape=input_shapes["Branch_IC_time"]["general"],
                     name = "Input-Branch1")
    z1 = cnn_model(input_b1)
    output_b1 = Dense(output_shapes["Out1"]["general"][0],\
                          activation="relu",\
                          name="Target1")(z1)
    #output_b2 = Dense(output_shapes["Out2"]["general"][0],\
    #                      activation="relu",\
    #                      name="Target2")(z1)
    #output_b3 = Dense(output_shapes["Out3"]["general"][0],\
    #                      activation="relu",\
    #                      name="Target3")(z1)
    #output_b4 = Dense(output_shapes["Out4"]["general"][0],\
    #                      activation="relu",\
    #                      name="Target4")(z1)


    model = keras.models.Model(inputs=[input_b1],
                               outputs=[output_b1])
    print(model.summary())
    return model

