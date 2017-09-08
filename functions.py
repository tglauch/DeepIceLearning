#!/usr/bin/env python
# coding: utf-8

"""This file is part of DeepIceLearning
DeepIceLearning is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D,\
    BatchNormalization, MaxPooling2D, Convolution3D, MaxPooling3D, Merge
from keras import regularizers
import os
import numpy as np
import tables
import resource


def gpu_memory():
    out = os.popen("nvidia-smi").read()
    ret = '0MiB'
    for item in out.split("\n"):
        if str(os.getpid()) in item and 'python' in item:
            ret = item.strip().split(' ')[-2]
    return float(ret[:-3])


class MemoryCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, log={}):
        print(' \n RAM Usage {:.2f} GB \n \n'.format(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
        os.system("nvidia-smi")


def read_input_len_shapes(file_location, input_files, virtual_len=-1):

    """Read length and shape attribute form
    datasets and assert that all files have been processed with
    the same detector shape. Return the length for further processing.

    Arguments:
    file_location: location of this file, as saved in config.cfg
    input_shape : The file list as above
    virtual_len : can be set for debugging purposes if only the
    first $virtual_len events shell be considered

    Returns:

    file_len : List of number of events for each file

    """
    file_len = []
    for run, input_file in enumerate(input_files):
        data_file = os.path.join(file_location, '{}'.format(input_file))
        file_handler = tables.open_file(data_file, 'r')
        if run == 0:
            test_shape = file_handler.root._v_attrs.shape
        else:
            if file_handler.root._v_attrs.shape != test_shape:
                raise Exception(
                    'The input files arrays do not have the same shape')
        if virtual_len == -1:
            data_len = file_handler.root._v_attrs.len
        else:
            data_len = virtual_len
            print('Only use the first {} Monte Carlo Events'.format(data_len))
        file_len.append(data_len)
        file_handler.close()
    return file_len


def generator(batch_size, file_handlers, inds,
              inp_shape_dict, inp_transformations,
              out_shape_dict, out_transformations, val_run=False):

    """ This function is a real braintwister and presumably really bad implemented.
    It produces all input and output data and applies the transformations
    as defined in the network definition file.

    Arguments:
    batch size : the batch size per gpu
    file_location: path to the folder containing the training files
    file_list: list of files used for the training
    inds: the index range used for the training set
    inp_shape_dict: A dictionary with the input shape for each branch
    inp_transformations: Dictionary with input variable name and function
    out_shape_dict: A dictionary with the output shape for each branch
    out_transformations: Dictionary with out variable name and function

    Returns:
    batch_input : a batch of input data
    batch_out: a batch of output data

    """

    in_branches = [(branch, inp_shape_dict[branch]['general'])
                   for branch in inp_shape_dict]
    out_branches = [(branch, out_shape_dict[branch]['general'])
                    for branch in out_shape_dict]
    inp_variables = [[(i, inp_transformations[branch[0]][i])
                      for i in inp_transformations[branch[0]]]
                     for branch in in_branches]
    out_variables = [[(i, out_transformations[branch[0]][i])
                      for i in out_transformations[branch[0]]]
                     for branch in out_branches]
    batch_input = [np.zeros((batch_size,) + branch[1])
                   for branch in in_branches]
    batch_out = [np.zeros((batch_size,) + (branch[1],))
                 for branch in out_branches]

    cur_file = 0
    cur_event_id = inds[cur_file][0]
    cur_len = 0
    up_to = inds[cur_file][1]
    loop_counter = 0
    temp_out = []
    temp_in = []
    while True:
        loop_counter += 1
        for j, var_array in enumerate(inp_variables):
            for k, var in enumerate(var_array):
                temp_cur_file = cur_file
                temp_cur_event_id = cur_event_id
                temp_up_to = up_to
                cur_len = 0
                while cur_len < batch_size:
                    fill_batch = batch_size - cur_len
                    if fill_batch < (temp_up_to - temp_cur_event_id):
                        if j == 0 and k == 0:
                            temp_out.extend(
                                file_handlers[cur_file]['reco_vals']
                                [temp_cur_event_id:
                                 temp_cur_event_id + fill_batch])
                        temp_in.extend(
                            eval('file_handlers[cur_file][\'{}\']'.
                                 format(var[0]))
                            [temp_cur_event_id:temp_cur_event_id + fill_batch])
                        cur_len += fill_batch
                        temp_cur_event_id += fill_batch
                    else:
                        temp_in.extend(
                            eval('file_handlers[cur_file][\'{}\']'.
                                 format(var[0]))
                            [temp_cur_event_id:temp_up_to])
                        if j == 0 and k == 0:
                            temp_out.extend(
                                file_handlers[cur_file]['reco_vals']
                                [temp_cur_event_id:temp_up_to])
                        cur_len += temp_up_to - temp_cur_event_id
                        temp_cur_file += 1
                        if temp_cur_file == len(file_handlers):
                            break
                        else:
                            temp_cur_event_id = inds[temp_cur_file][0]
                            temp_up_to = inds[temp_cur_file][1]

                for i in range(len(temp_in)):
                    slice_ind = [slice(None)] * batch_input[j][i].ndim
                    slice_ind[-1] = slice(k, k + 1, 1)
                    pre_append = var[1](temp_in[i])
                    if var == 'time':
                        pre_append[pre_append == np.inf] = -1
                    batch_input[j][i][slice_ind] = pre_append
                temp_in = []

        for j, var_array in enumerate(out_variables):
            for k, var in enumerate(var_array):
                for i in range(len(temp_out)):
                    batch_out[j][i][k] = var[1](temp_out[i][var[0]])
            temp_out = []

        if temp_cur_file == len(file_handlers):
            cur_file = 0
            cur_event_id = inds[0][0]
            up_to = inds[0][1]
        else:
            if temp_cur_file != cur_file:
                print('\n Read File Number {} \n'.format(temp_cur_file + 1))
                if not val_run:
                    print(' \n CPU RAM Usage {:.2f} GB'.
                          format(resource.getrusage(
                                 resource.RUSAGE_SELF).ru_maxrss / 1e6))
                    print(' GPU MEM : {:.2f} GB \n \n'.
                          format(gpu_memory() / 1e3))
            cur_file = temp_cur_file
            cur_event_id = temp_cur_event_id
            up_to = temp_up_to
        yield (batch_input, batch_out)


def read_NN_weights(args_dict, model):

    if args_dict['load_weights'] != 'None':
        print('Load Weights from {}'.format(args_dict['load_weights']))
        model.load_weights(args_dict['load_weights'])

    elif args_dict['continue'] != 'None':
        read_from = os.path.join(args_dict['continue'], 'best_val_loss.npy')
        print('Load Weights from {}'.format(read_from))
        model.load_weights(read_from)

    else:
        print('Initalize the model without pre-trained weights')

    return model


def residual_unit(x0):

    x = Convolution3D(64, (1, 1, 1), border_mode="same", activation="relu")(x0)
    x = Convolution3D(64, (3, 3, 3), border_mode="same", activation="relu")(x0)
    return Merge.add([x, x0])


def dense_block(x0, n=8):
    """ Create a block of n densely connected pairs of convolutions """
    for i in range(n):
        x = Convolution3D(8, (3, 3, 3), padding='same', activation='relu')(x0)
        x0 = Merge.concatenate([x0, x], axis=-1)
    return x


def inception_unit(x0):
    x1 = Convolution3D(16, (1, 1, 1), padding='same', activation='relu')(x0)
    x1 = Convolution3D(16, (3, 3, 3), padding='same', activation='relu')(x1)

    x2 = Convolution3D(16, (1, 1, 1), padding='same', activation='relu')(x0)
    x2 = Convolution3D(16, (5, 5, 5), padding='same', activation='relu')(x2)

    x3 = Convolution3D(16, (1, 1, 1), padding='same', activation='relu')(x0)

    x4 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x0)
    x4 = Convolution3D(64, (1, 1, 1), padding='same', activation='relu')(x4)


    return Merge.concatenate([x1, x2, x3, x4], axis=1)
