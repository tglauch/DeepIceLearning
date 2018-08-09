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
sys.path.append("/scratch9/mkron/software/DeepIceLearning/lib")
import transformations as tr
import numpy as np
import block_units as bunit


# *Settings*
# define inputs for each branch
inputs = OrderedDict()


#Input for IC
inputs["Branch_IC_charge"] = {"variables": ["IC_charge", "IC_first_charge", "IC_num_pulses", "IC_time_first"],
                     "transformations": [tr.centralize, tr.centralize,  tr.centralize, tr.centralize]}
inputs["Branch_IC_charge_abs"] = {"variables": ["IC_charge"],
                     "transformations": [tr.log_of_sum]}

inputs["Branch_IC_time"] = {"variables": ["IC_time_first", "IC_ATWD_0_05_pct_charge_quantile", "IC_ATWD_0_1_pct_charge_quantile",\
                                   "IC_ATWD_0_15_pct_charge_quantile", "IC_ATWD_0_2_pct_charge_quantile", "IC_ATWD_0_25_pct_charge_quantile",\
                                   "IC_ATWD_0_3_pct_charge_quantile", "IC_ATWD_0_35_pct_charge_quantile", "IC_ATWD_0_4_pct_charge_quantile",\
                                   "IC_ATWD_0_45_pct_charge_quantile", "IC_ATWD_0_5_pct_charge_quantile", "IC_ATWD_0_55_pct_charge_quantile",\
                                   "IC_ATWD_0_6_pct_charge_quantile", "IC_ATWD_0_65_pct_charge_quantile", "IC_ATWD_0_7_pct_charge_quantile",\
                                   "IC_ATWD_0_75_pct_charge_quantile","IC_ATWD_0_8_pct_charge_quantile", "IC_ATWD_0_85_pct_charge_quantile",\
                                   "IC_ATWD_0_9_pct_charge_quantile","IC_ATWD_0_95_pct_charge_quantile", "IC_charge"],
                     "transformations": [tr.centralize, tr.centralize, tr.centralize,\
                                         tr.centralize, tr.centralize, tr.centralize,\
                                         tr.centralize, tr.centralize, tr.centralize,\
                                         tr.centralize, tr.centralize, tr.centralize,\
                                         tr.centralize, tr.centralize, tr.centralize,\
                                         tr.centralize, tr.centralize, tr.centralize,\
                                         tr.centralize, tr.centralize, tr.centralize]}
inputs["Branch_IC_time_abs"] = {"variables": ["IC_time_first"],
                     "transformations": [tr.max_min_delta_log]}

#Input for DC
inputs["Branch_DC_charge"] = {"variables": ["DC_charge", "DC_first_charge", "DC_num_pulses", "DC_time_first"],
                     "transformations": [tr.centralize, tr.centralize,  tr.centralize, tr.centralize]}
inputs["Branch_DC_charge_abs"] = {"variables": ["DC_charge"],
                     "transformations": [tr.log_of_sum]}

inputs["Branch_DC_time"] = {"variables": ["DC_time_first", "DC_ATWD_0_05_pct_charge_quantile", "DC_ATWD_0_1_pct_charge_quantile",\
                                   "DC_ATWD_0_15_pct_charge_quantile", "DC_ATWD_0_2_pct_charge_quantile", "DC_ATWD_0_25_pct_charge_quantile",\
                                   "DC_ATWD_0_3_pct_charge_quantile", "DC_ATWD_0_35_pct_charge_quantile", "DC_ATWD_0_4_pct_charge_quantile",\
                                   "DC_ATWD_0_45_pct_charge_quantile", "DC_ATWD_0_5_pct_charge_quantile", "DC_ATWD_0_55_pct_charge_quantile",\
                                   "DC_ATWD_0_6_pct_charge_quantile", "DC_ATWD_0_65_pct_charge_quantile", "DC_ATWD_0_7_pct_charge_quantile",\
                                   "DC_ATWD_0_75_pct_charge_quantile","DC_ATWD_0_8_pct_charge_quantile", "DC_ATWD_0_85_pct_charge_quantile",\
                                   "DC_ATWD_0_9_pct_charge_quantile","DC_ATWD_0_95_pct_charge_quantile", "DC_charge"],
                     "transformations": [tr.centralize, tr.centralize, tr.centralize,\
                                         tr.centralize, tr.centralize, tr.centralize,\
                                         tr.centralize, tr.centralize, tr.centralize,\
                                         tr.centralize, tr.centralize, tr.centralize,\
                                         tr.centralize, tr.centralize, tr.centralize,\
                                         tr.centralize, tr.centralize, tr.centralize,\
                                         tr.centralize, tr.centralize, tr.centralize]}
inputs["Branch_DC_time_abs"] = {"variables": ["DC_time_first"],
                     "transformations": [tr.max_min_delta_log]}
                     
# define outputs for each branch
outputs = OrderedDict()
outputs["Out1"] = {"variables": ["ClassificationLabel"],
#                   "transformations": [tr.oneHotEncode_EventType]}
                   "transformations": [tr.oneHotEncode_EventType_stratingTrack]}

outputs["Out2"] = {"variables": ["StartingLabel"],
                   "transformations": [tr.oneHotEncode_01]}

outputs["Out3"] = {"variables": ["zenith"],
                   "transformations": [tr.zenith_prep]}

reference_outputs = []

# *Model*

def model(input_shapes, output_shapes):

    kwargs = dict(activation='relu', kernel_initializer='he_normal')

    #IC
    #branch1
    input_b1 = Input(shape=input_shapes["Branch_IC_charge"]["general"],
                     name = "Input-Branch1")
    z1 = Conv3D(72, (3, 3, 5), padding="same", **kwargs)(input_b1)
    z1 = MaxPooling3D(pool_size=(2, 2, 3))(z1)
    z1 = BatchNormalization()(z1)
    z1 = bunit.Residual(72, 72, z1)
    z1 = BatchNormalization()(z1)
    z1 = bunit.Residual(72, 72, z1)
    z1 = BatchNormalization()(z1)
    z1 = bunit.Residual(72, 72, z1)
    z1 = BatchNormalization()(z1)
    z1 = bunit.Residual(72, 32, z1)
    z1 = BatchNormalization()(z1)
    z1 = Flatten()(z1)
    z1 = Dense(64, **kwargs)(z1)
    z1 = Dropout(rate=0.4)(z1)


    # branch 2
    input_b2 = Input(shape=input_shapes["Branch_IC_charge_abs"]["general"],
                     name="Input-Branch2")

    # merge #1 of Branche "IC_charge"
    merge1 = concatenate([z1, input_b2])
    merge1 = Dense(36,\
              **kwargs)(merge1)
    merge1 = Dropout(rate=0.4)(merge1)
    merge1 = BatchNormalization()(merge1)


    #branch3
    input_b3 = Input(shape=input_shapes["Branch_IC_time"]["general"],
                     name = "Input-Branch3")
    z3 = Conv3D(72, (3, 3, 5), padding="same", **kwargs)(input_b3)
    z3 = MaxPooling3D(pool_size=(2, 2, 3))(z3)
    z3 = BatchNormalization()(z3)
    z3 = bunit.Residual(72, 72, z3)
    z3 = BatchNormalization()(z3)
    z3 = bunit.Residual(72, 72, z3)
    z3 = BatchNormalization()(z3)
    z3 = bunit.Residual(72, 72, z3)
    z3 = BatchNormalization()(z3)
    z3 = bunit.Residual(72, 32, z3)
    z3 = BatchNormalization()(z3)
    z3 = Flatten()(z3)
    z3 = Dense(64, **kwargs)(z3)
    z3 = Dropout(rate=0.4)(z3)

    # branch 4 
    input_b4 = Input(shape=input_shapes["Branch_IC_time_abs"]["general"],
                     name="Input-Branch4")

    # merge #2 of Branche "IC_time"
    merge2 = concatenate([z3, input_b4])
    merge2 = Dense(72,\
              **kwargs)(merge2)
    merge2 = Dropout(rate=0.4)(merge2)
    merge2 = BatchNormalization()(merge2)

    # merge total IC
    merge_IC = concatenate([merge1, merge2])
    merge_IC = bunit.Dense_Residual(108, 72, merge_IC)  # #input shape has to match the previous layer
    merge_IC = bunit.Dense_Residual(72, 72, merge_IC)
    merge_IC = bunit.Dense_Residual(72, 72, merge_IC)
    merge_IC = bunit.Dense_Residual(72, 36, merge_IC)

    #DC
    #branch5
    input_b5 = Input(shape=input_shapes["Branch_DC_charge"]["general"],
                     name = "Input-Branch5")
    z5 = Conv3D(72, (3, 3, 5), padding="same", **kwargs)(input_b5)
    z5 = MaxPooling3D(pool_size=(2, 2, 3))(z5)
    z5 = BatchNormalization()(z5)
    z5 = bunit.Residual(72, 72, z5)
    z5 = BatchNormalization()(z5)
    z5 = bunit.Residual(72, 72, z5)
    z5 = BatchNormalization()(z5)
    z5 = bunit.Residual(72, 72, z5)
    z5 = BatchNormalization()(z5)
    z5 = bunit.Residual(72, 32, z5)
    z5 = BatchNormalization()(z5)
    z5 = Flatten()(z5)
    z5 = Dense(64, **kwargs)(z5)
    z5 = Dropout(rate=0.4)(z5)


    # branch 6
    input_b6 = Input(shape=input_shapes["Branch_DC_charge_abs"]["general"],
                     name="Input-Branch6")

    # merge #3 of Branche "DC_charge"
    merge3 = concatenate([z5, input_b6])
    merge3 = Dense(36,\
              **kwargs)(merge3)
    merge3 = Dropout(rate=0.4)(merge3)
    merge3 = BatchNormalization()(merge3)


    #branch7
    input_b7 = Input(shape=input_shapes["Branch_DC_time"]["general"],
                     name = "Input-Branch7")
    z7 = Conv3D(72, (3, 3, 5), padding="same", **kwargs)(input_b7)
    z7 = MaxPooling3D(pool_size=(2, 2, 3))(z7)
    z7 = BatchNormalization()(z7)
    z7 = bunit.Residual(72, 72, z7)
    z7 = BatchNormalization()(z7)
    z7 = bunit.Residual(72, 72, z7)
    z7 = BatchNormalization()(z7)
    z7 = bunit.Residual(72, 72, z7)
    z7 = BatchNormalization()(z7)
    z7 = bunit.Residual(72, 32, z7)
    z7 = BatchNormalization()(z7)
    z7 = Flatten()(z7)
    z7 = Dense(64, **kwargs)(z7)
    z7 = Dropout(rate=0.4)(z7)

    # branch 8
    input_b8 = Input(shape=input_shapes["Branch_DC_time_abs"]["general"],
                     name="Input-Branch8")

    # merge #4 of Branche "DC_time"
    merge4 = concatenate([z7, input_b8])
    merge4 = Dense(72,\
              **kwargs)(merge4)
    merge4 = Dropout(rate=0.4)(merge4)
    merge4 = BatchNormalization()(merge4)


    # merge total DC
    merge_DC = concatenate([merge3, merge4])
    merge_DC = bunit.Dense_Residual(108, 72, merge_DC)  # #input shape has to match the previous layer
    merge_DC = bunit.Dense_Residual(72, 36, merge_DC)

    # merge total total
    merge_total = concatenate([merge_IC, merge_DC])
    merge_total = bunit.Dense_Residual(72, 72, merge_total)  # #input shape has to match the previous layer
    merge_total = bunit.Dense_Residual(72, 72, merge_total)
    merge_total = bunit.Dense_Residual(72, 72, merge_total)
    merge_total = bunit.Dense_Residual(72, 36, merge_total)

    # output 1
    o1 = bunit.Dense_Residual(36, 36, merge_total)
    o1 = bunit.Dense_Residual(36, 36, o1)
    o1 = bunit.Dense_Residual(36, 36, o1)
    o1 = bunit.Dense_Residual(36, 36, o1)
    o1 = bunit.Dense_Residual(36, 36, o1)
    o1 = bunit.Dense_Residual(36, 36, o1)
    o1 = bunit.Dense_Residual(36, 36, o1)
    o1 = bunit.Dense_Residual(36, 36, o1)
    o1 = bunit.Dense_Residual(36, 36, o1)
    output_b1 = Dense(output_shapes["Out1"]["general"][0],\
                          activation="softmax",\
                          name="Target1")(o1)


    # output 2
    o2 = bunit.Dense_Residual(36, 36, merge_total)
    o2 = bunit.Dense_Residual(36, 36, o2)
    o2 = bunit.Dense_Residual(36, 36, o2)
    output_b2 = Dense(output_shapes["Out2"]["general"][0],\
                          activation="softmax",\
                          name="Target2")(o2)


    # output 3
    o3 = bunit.Dense_Residual(36, 36, merge_total)
    o3 = bunit.Dense_Residual(36, 36, o3)
    o3 = bunit.Dense_Residual(36, 36, o3)
    output_b3 = Dense(output_shapes["Out3"]["general"][0],\
                          activation="sigmoid",\
                          name="Target3")(o3)


    model = keras.models.Model(inputs=[input_b1, input_b2, input_b3, input_b4, input_b5, input_b6, input_b7, input_b8],\
                               outputs=[output_b1, output_b2, output_b3])
    print(model.summary())
    return model

