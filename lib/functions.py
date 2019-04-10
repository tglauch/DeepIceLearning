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
from keras.layers import *
import os
import numpy as np
import tables
import resource
import h5py
from keras.callbacks import Callback
import time

class WeightsSaver(Callback):
    def __init__(self, N, save_path):
        self.N = N
        self.batch = 0
        self.save_path = save_path

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = os.path.join(self.save_path, "model_all_epochs/batch/weights_batch%06d.npy" % self.batch)
            self.model.save_weights(name)
        self.batch += 1

class ParallelModelCheckpoint(keras.callbacks.ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
		self.single_model = model
		super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)

def close_h5file(file_obj):
    if isinstance(file_obj, h5py.File):   # Just HDF5 files
        try:
            file_obj.close()
        except:
            pass
    return

def read_variables(cfg_parser):
    """Function reading a config file, defining the variables to be read
       from the MC files.

    Arguments:
    cfg_parser: config parser object for the config file

    Returns:
    dtype : the dtype object defining the shape and names of the MC output
    data_source: list defining the types,names and ranges of monte carlo data
                to be saved from a physics frame
                (e.g [('variable',['MCMostEnergeticTrack'].energy, [1e2,1e9])])
    """
    dtype = []
    data_source = []
    for i, key in enumerate(cfg_parser.keys()):
        if key == 'DEFAULT' or key == 'Basics' or key =='Cuts' or key =='Scale_Class'\
           or 'Input' in key:
            continue
        cut = [-np.inf, np.inf]
        if 'min' in cfg_parser[key].keys():
            cut[0] = float(cfg_parser[key]['min'])
        if 'max' in cfg_parser[key].keys():
            cut[1] = float(cfg_parser[key]['max'])
        if 'variable' in cfg_parser[key].keys():
            data_source.append(('variable', cfg_parser[key]['variable'], cut))
        elif 'function' in cfg_parser[key].keys():
            data_source.append(('function', cfg_parser[key]['function'], cut))
        else:
            raise Exception(
                'No Input Type given. Variable or funtion must be given')
        dtype.append((str(key), eval('np.' + cfg_parser[key]['out_type'])))
    dtype = np.dtype(dtype)

    return dtype, data_source


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
        if not os.path.isabs(input_file):
            data_file = os.path.join(file_location, input_file)
        else:
            data_file = input_file
        file_handler = tables.open_file(data_file, 'r')
        if run == 0:
            test_shape = file_handler.root._v_attrs.shape
        else:
            if file_handler.root._v_attrs.shape != test_shape:
                raise Exception(
                    'The input files arrays do not have the same shape')
        if virtual_len == -1:
            data_len = len(file_handler.root.reco_vals)
        else:
            data_len = virtual_len
            print('Only use the first {} Monte Carlo Events'.format(data_len))
        file_len.append(data_len)
        file_handler.close()
    return file_len


def generator_v2(batch_size, file_handlers, inds, inp_shape_dict,
                 inp_transformations,out_shape_dict, out_transformations,
                 weighting_function=None, use_data=False):

    """ This function is a real braintwister and presumably really bad implemented.
    It produces all input and output data and applies the transformations
    as defined in the network definition file.

    Arguments:
    batch size : the batch size per gpu
    file_handlers: list of files used for the training
    inds: the index range used for the dataset
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
    cur_file = 0
    ind_lo = inds[0][0]
    ind_hi = inds[0][0] + batch_size
    in_data = h5py.File(file_handlers[0])
    while True:
        t0 = time.time()
        inp_data = []
        out_data = []
        arr_size = np.min([batch_size, ind_hi - ind_lo])
        # Generate Input Data  
        for k, b in enumerate(in_branches):
            batch_input = np.zeros((arr_size,)+in_branches[k][1])
            for j, f in enumerate(inp_variables[k]):
                if f[0] in in_data.keys():
                    pre_data = np.squeeze(in_data[f[0]][ind_lo:ind_hi])
                    batch_input[:,:,:,:,j] = f[1](pre_data)
                else:
                    pre_data = np.squeeze(in_data['reco_vals'][f[0]][ind_lo:ind_hi])
                    batch_input[:,j]=f[1](pre_data)
            inp_data.append(batch_input)
            
        # Generate Output Data
        for k, b in enumerate(out_branches):
            batch_output = np.zeros((arr_size,)+out_branches[k][1])
            for j, f in enumerate(out_variables[k]):
                pre_data = np.squeeze(in_data['reco_vals'][f[0]][ind_lo:ind_hi])
                batch_output[:,j]=f[1](pre_data)
            out_data.append(batch_output)


        #Prepare next round
        ind_lo += batch_size
        ind_hi += batch_size
        if ind_lo >= inds[cur_file][1]:
            cur_file += 1
            if cur_file == len(file_handlers):
                cur_file=0
            print('Open File {}'.format(file_handlers[cur_file]))
            in_data.close()
            in_data = h5py.File(file_handlers[cur_file])
            ind_lo = inds[cur_file][0]
            ind_hi = ind_lo + batch_size
        elif ind_hi > inds[cur_file][1]:
            ind_hi = inds[cur_file][1]

        # Yield Result
        t1 = time.time()
        print('\n generate batch in {}s '.format((t1-t0)))
        if use_data:
            yield inp_data
        else:
            yield (inp_data, out_data)


def generator(batch_size, file_handlers, inds,
              inp_shape_dict, inp_transformations,
              out_shape_dict, out_transformations, use_data=False):

    """ This function is a real braintwister and presumably really bad implemented.
    It produces all input and output data and applies the transformations
    as defined in the network definition file.

    Arguments:
    batch size : the batch size per gpu
    file_location: path to the folder containing the training files
    file_list: list of files used for the training
    inds: the index range used for the dataset
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
    batch_out = [np.zeros((batch_size,) + branch[1])
                 for branch in out_branches]
    cur_file = 0
    cur_event_id = inds[cur_file][0]
    cur_len = 0
    up_to = inds[cur_file][1]
    loop_counter = 0
    temp_out = []
    temp_in = []
    t_file = None
    while True:
        loop_counter += 1
        t0 = time.time()
        for j, var_array in enumerate(inp_variables):
            for k, var in enumerate(var_array):
                temp_cur_file = cur_file
                close_h5file(t_file)
                t_file = h5py.File(file_handlers[temp_cur_file], 'r')
                temp_cur_event_id = cur_event_id
                temp_up_to = up_to
                cur_len = 0
                while cur_len < batch_size:
                    fill_batch = batch_size - cur_len
                    if fill_batch < (temp_up_to - temp_cur_event_id):
                        if var[0] in t_file.keys():
                            temp_in.extend(
                                t_file[var[0]]
                                [temp_cur_event_id:temp_cur_event_id + fill_batch])
                        else:
                            temp_in.extend(
                                t_file['reco_vals'][var[0]]
                                [temp_cur_event_id:temp_cur_event_id + fill_batch])
                        cur_len += fill_batch
                        temp_cur_event_id += fill_batch
                    else:
                        if var[0] in t_file.keys():
                            temp_in.extend(
                                t_file[var[0]]
                                [temp_cur_event_id:temp_up_to])
                        else:
                            temp_in.extend(
                                t_file['reco_vals'][var[0]]
                                [temp_cur_event_id:temp_up_to])
                        cur_len += temp_up_to - temp_cur_event_id
                        t_file.close()
                        temp_cur_file += 1
                        if temp_cur_file == len(file_handlers):
                            break
                        else:
                            t_file = h5py.File(file_handlers[temp_cur_file], 'r')
                            temp_cur_event_id = inds[temp_cur_file][0]
                            temp_up_to = inds[temp_cur_file][1]
                for i in range(len(temp_in)):
                    slice_ind = [slice(None)] * batch_input[j][i].ndim
                    slice_ind[-1] = slice(k, k + 1, 1)
                    pre_append = var[1](temp_in[i])
                    if len(var_array) > 1:
                        batch_input[j][i][slice_ind] = pre_append
                    else:
                        batch_input[j][i] = pre_append
                temp_in = []
        for j, var_array in enumerate(out_variables):
            if use_data:
                continue
            for k, var in enumerate(var_array):
                temp_cur_file = cur_file
                close_h5file(t_file)
                t_file = h5py.File(file_handlers[temp_cur_file], 'r')
                if j==0 and k==0:
                    print('\n Open File (1) {}'.format(file_handlers[temp_cur_file]))
                temp_cur_event_id = cur_event_id
                temp_up_to = up_to
                cur_len = 0
                event_list = []
                while cur_len < batch_size:
                    fill_batch = batch_size - cur_len
                    if fill_batch < (temp_up_to - temp_cur_event_id):
                        temp_out.extend(
                            t_file['reco_vals']
                            [var[0]][temp_cur_event_id:temp_cur_event_id +
                                     fill_batch])
                        event_list.extend(zip(np.full(fill_batch, temp_cur_file),
                                              range(temp_cur_event_id,
                                                    temp_cur_event_id +
                                                    fill_batch)))
                        cur_len += fill_batch
                        temp_cur_event_id += fill_batch
                    else:
                        temp_out.extend(
                            t_file['reco_vals'][var[0]]
                            [temp_cur_event_id:temp_up_to])
                        event_list.extend(zip(np.full(temp_up_to -
                                                      temp_cur_event_id,
                                                      temp_cur_file),
                                              range(temp_cur_event_id,
                                                    temp_up_to)))
                        cur_len += temp_up_to - temp_cur_event_id
                        t_file.close()
                        temp_cur_file += 1
                        if temp_cur_file == len(file_handlers):
                            break
                        else:
                            t_file = h5py.File(file_handlers[temp_cur_file], 'r')
                            if j==0 and k==0:
                                print('Open File (2) {}'.format(file_handlers[temp_cur_file]))
                            temp_cur_event_id = inds[temp_cur_file][0]
                            temp_up_to = inds[temp_cur_file][1]
                t_file.close()
                file_counter = event_list[0][0]
                t_file = h5py.File(file_handlers[file_counter], 'r')
                for i in range(len(temp_out)):
                    if event_list[i][0] != file_counter:
                        t_file.close()
                        file_counter = event_list[i][0]
                        t_file = h5py.File(file_handlers[file_counter], 'r')
                    slice_ind = [slice(None)] * batch_out[j][i].ndim
                    slice_ind[-1] = slice(k, k + 1, 1)
                    pre_append = var[1](temp_out[i],
                                        t_file['reco_vals'][:][event_list[i][1]])
                    if var == 'time':
                        pre_append[pre_append == np.inf] = -1
                    if len(var_array) > 1:
                        batch_out[j][i][slice_ind] = pre_append
                    else:
                        batch_out[j][i] = pre_append
                temp_out = []
                t_file.close()

        if temp_cur_file == len(file_handlers):
            cur_file = 0
            cur_event_id = inds[0][0]
            up_to = inds[0][1]
        else:
            if temp_cur_file != cur_file:
                print(' \n CPU RAM Usage {:.2f} GB'.
                      format(resource.getrusage(
                             resource.RUSAGE_SELF).ru_maxrss / 1e6))
                print(' GPU MEM : {:.2f} GB \n \n'.
                      format(gpu_memory() / 1e3))
            cur_file = temp_cur_file
            cur_event_id = temp_cur_event_id
            up_to = temp_up_to
        t1 = time.time()
        print('Time for one loop {}s'.format(t1-t0))
        print(type(batch_input))
        print(np.shape(batch_input))
        print(type(batch_out))
        print(np.shape(batch_out))
        if use_data:
            yield batch_input
        else:
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
