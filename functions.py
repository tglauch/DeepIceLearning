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
 BatchNormalization, MaxPooling2D,Convolution3D,MaxPooling3D, Merge
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
        file_handler = tables.openFile(data_file, 'r')
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


def prepare_input_output_variables(file_path, model_settings):

    """given the transformations of the 'raw' input data
    (e.g. the charges per gridelement) the
    the input shape for the network is calculated.
    example: transformation is np.sum(x) -->
    input shape = (1,)


    Arguments:
    file_path : the path to one exemplary input file
    model_settings : the different settings for the model.
    has to include a section [Inputs] otherwise assume no transformation at all

    Returns:
    shapes : the input shapes for each branch after transformation (ordered)

    shape_names : the names of the corresponding model branches (ordered)

    inp_variables : list of the input variables for the different branches
     (e.g [['charge','time'], ['charge']])

    inp_transformation : the corresponding transformations applied
    before feeding the network (e.g [['x-np.mean(x)','x'], ['np.sum(x)']])

    out_variables: list of output variables (e.g. ['energy', 'zenith'])

    out_transformation: list of output transformations
    (e.g. ['np.log10(x)', 'x'])
    """

    inp_variables = []
    inp_transformations = []
    out_variables = []
    out_transformations = []
    shapes = []
    shape_names = []
    mode = ''
    if len(model_settings) == 0:
        model_settings = [['{Inputs}'],
                          ['[model]', 'variables = [charge]',
                          'transformations = [x]']]
    for block in model_settings:
        if block[0][0] == '{' and block[0][-1] == '}':
            mode = block[0][1:-1]
            continue
        if mode == 'Inputs':
            for element in block:
                if element[0] == '[' and element[-1] == ']':
                    shape_names.append(element[1:-1])
                else:
                    split_row = element.split('=')
                    if split_row[0].strip() == 'variables':
                        inp_variables.append(
                            [unstripped.strip() for unstripped
                                in split_row[1].split(',')])
                    elif split_row[0].strip() == 'transformations':
                        inp_transformations.append(
                            [unstripped.strip() for unstripped
                                in split_row[1].split(',')])
        elif mode == 'Outputs':
            for element in block:
                split_row = element.split('=')
                if split_row[0].strip() == 'variables':
                    out_variables.extend(split_row[1].strip().split(','))
                elif split_row[0].strip() == 'transformations':
                    out_transformations.extend(split_row[1].strip().split(','))
        else:
            print('No idea what to do...:(....continue')
            continue

    cur_file_handler = tables.openFile(file_path)
    for i in range(len(shape_names)):
        temp_shape_arr = []
        for j in range(len(inp_variables[i])):
            test_array = eval(
                'cur_file_handler.root.{}'.format(inp_variables[i][j]))[0]
            temp_shape_arr.append(
                np.shape(eval(
                    inp_transformations[i][j].replace('x', 'test_array'))))
        if all(x == temp_shape_arr[0] for x in temp_shape_arr):
            shapes.append(temp_shape_arr[0][0:-1] + (len(temp_shape_arr),))
        else:
            raise Exception(
                'The transformations that you have applied do not results\
                 in the same shape!!!!')
    cur_file_handler.close()
    return shapes, shape_names, inp_variables,\
        inp_transformations, out_variables, out_transformations


def calc_depositedE(physics_frame):
    I3Tree = physics_frame['I3MCTree']
    truncated_energy = 0
    for i in I3Tree:
        interaction_type = str(i.type)
        if interaction_type in ['DeltaE', 'PairProd', 'Brems', 'EMinus']:
            truncated_energy += i.energy
    return truncated_energy


def parse_config_file(conf_file_path):
    """Function that parses the config file and
    returns the settings and architecture of the model

    Arguments:
    conf_file_path : name of the config file in the folder

    Returns:
    settings : settings for the network. e.g. the input shapes
    model : the 'architecture' of the model. basically contains a
    list of layers with their args and kwargs
    """

    f = open(conf_file_path)
    config_array = f.read().splitlines()
    config_blocks = []
    single_block = []
    for line in config_array:
        if line == '':
            config_blocks.append(single_block)
            single_block = []
        else:
            single_block.append(line)
    settings = []
    model = []
    mode = ''
    for block in config_blocks:
        if mode == '' or block[0][0] == '*':
            if block[0] == '*Settings*':
                mode = 'settings'
            elif block[0] == '*Model*':
                mode = 'model'
            else:
                raise Exception('config file is corrupted')
        else:
            if mode == 'settings':
                settings.append(block)
            elif mode == 'model':
                model.append(block)
    return settings, model


def add_layer(model, layer, args, kwargs):

    """Given the data read from the network configuration
    file, add a layer to the Keras xnetwork model object

    Arguments:
    model : the model object of the network
    layer (str): the type of layer (https://keras.io/layers/core/)

    Returns: True

    """
    eval('model.add({}(*args,**kwargs))'.format(layer))
    return


def base_model(model_def, shapes, shape_names):
    """Main function to create the Keras Neural Network.

    Arguments:
    model_def : list containing blocks (list) of model-branches
    and layer definitions (compare the network config files)
    shape, shape_names: input shapes and names as constructed
    in prepare_input_output_variables(file_path, model_settings)

    Returns:
    model : the (non-compiled) model object

    """
    models = dict()
    cur_model = None
    cur_model_name = ''
    for block in model_def:
        if block[0][0] == '{' and block[0][-1] == '}' or cur_model is None:
            if cur_model is not None:
                print(cur_model.summary())
                models[cur_model_name] = cur_model
            cur_model = Sequential()
            input_layer = True
            if block[0][0] == '{' and block[0][-1] == '}':
                cur_model_name = block[0][1:-1]
            else:
                cur_model_name = 'model'
        if block[0][0] == '[' and block[0][-1] == ']':
            args = []
            kwargs = dict()
            layer = ''
            mode = 'args'
            layer = block[0][1:-1]
            for i in range(1, len(block)):
                if block[i] == '[kwargs]':
                    mode = 'kwargs'
                elif mode == 'args':
                    try:
                        args.append(eval(block[i].split('=')[1].strip()))
                    except Exception:
                        args.append(block[i].split('=')[1].strip())
                elif mode == 'kwargs':
                    split_line = block[i].split('=')
                    try:
                        kwargs[split_line[0].strip()] = eval(
                            split_line[1].strip())
                    except Exception:
                        kwargs[split_line[0].strip()] = split_line[1].strip()
            if not layer == 'Merge':
                if 'input_shape' not in kwargs and input_layer is True:
                    ind = shape_names.index(cur_model_name)
                    kwargs['input_shape'] = shapes[ind]
                add_layer(cur_model, layer, args, kwargs)
            else:
                merge_layer_names = [name.strip()
                                     for name in kwargs['layers']
                                     [1:-1].split(',')]
                kwargs = dict()
                kwargs['mode'] = 'concat'
                add_layer(
                    cur_model,
                    layer,
                    [[models[name] for name in merge_layer_names]],
                    kwargs)
                for name in merge_layer_names:
                    del models[name]
            input_layer = False
    print(cur_model.summary())
    models[cur_model_name] = cur_model
    return cur_model


class MemoryCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, log={}):
        print(' \n RAM Usage {:.2f} GB \n \n'.format(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
        os.system("nvidia-smi")


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
    inp_shape: the shapes of the input neurons
    inp_variables: list of variables used for the input
    inp_transformations: list of transformations for the input data
    out_variables: variables for the output data
    out_transformations: transformations applied to the output data

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
                            eval('file_handlers[cur_file][\'{}\']'.format(var[0]))
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
