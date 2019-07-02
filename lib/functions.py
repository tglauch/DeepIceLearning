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
		super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only,
                                                     save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)

class every_model(ParallelModelCheckpoint):
    def __init__(self, model, filepath, verbose,
                 monitor='val_loss',
                 save_best_only=False,
                 mode='auto',
                 period=1):
        self.verbose = verbose
        self.single_model = model

def chose_optimizer(optimizer, learningrate):
    if optimizer == "Nadam":
        print "Optimizer: Nadam"
        optimizer_used = keras.optimizers.Nadam(lr=learningrate)
    elif optimizer == "Adam":
        print "Optimizer: Adam"
        optimizer_used = keras.optimizers.Adam(lr=learningrate)
    elif optimizer == "SGD":
        print "Optimizer: SGD"
        optimizer_used = keras.optimizers.SGD(lr=learningrate)
    elif optimizer == "RMSProb":
        print "Optimizer: RMSProb"
        optimizer_used = keras.optimizers.RMSprob(lr=learningrate)
    else:
        print "Optimizer unchoosen or unknown -> default: Adam"
        optimizer_used = keras.optimizers.Adam(lr=learningrate)
    return optimizer_used

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
                 inp_transformations, out_shape_dict, out_transformations,
                 weighting_function=None, use_data=False, equal_len=False,
                 mask_func=None, valid=False):

    """ This function generates the training batches for the neural network.
    It load all input and output data and applies the transformations
    as defined in the network definition file.

    Arguments:
    batch size : the batch size per gpu
    file_handlers: list of files used for the training
                   i.e. ['/path/to/file/A', 'path/to/file/B']
    inds: the index range used for the dataset
          i.e. [(0,1000), (0,2000)]
    inp_shape_dict: A dictionary with the input shape for each branch
    inp_transformations: Dictionary with input variable name and function
    out_shape_dict: A dictionary with the output shape for each branch
    out_transformations: Dictionary with out variable name and function

    Returns:
    batch_input : a batch of input data
    batch_out: a batch of output data

    """

    print('Run with inds {}'.format(inds))
    file_handlers = np.array(file_handlers)
    inds = np.array(inds)
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
#    print inp_transformations
#    print out_transformations
#    print inp_shape_dict
#    print out_shape_dict
    cur_file = 0
    ind_lo = inds[0][0]
    ind_hi = inds[0][0] + batch_size
    in_data = h5py.File(file_handlers[0], 'r')
    f_reco_vals = in_data['reco_vals']
    t0 = time.time()
    num_batches = 1
 
    while True:
        inp_data = []
        out_data = []
        weights = []
        arr_size = np.min([batch_size, ind_hi - ind_lo])
        reco_vals = f_reco_vals[ind_lo:ind_hi]

        #print('Generate Input Data')
        for k, b in enumerate(out_branches):
            for j, f in enumerate(out_variables[k]):
                if weighting_function != None:
                    tweights=weighting_function(reco_vals)
                else:
                    tweights=np.ones(arr_size)
                if mask_func != None:
                    mask = mask_func(reco_vals)
                    tweights[mask] = 0
            weights.append(tweights)
            
        for k, b in enumerate(in_branches):
            batch_input = np.zeros((arr_size,)+in_branches[k][1])
            for j, f in enumerate(inp_variables[k]):
                if f[0] in in_data.keys():
                    pre_data = np.array(np.squeeze(in_data[f[0]][ind_lo:ind_hi]), ndmin=4)
                    batch_input[:,:,:,:,j] = np.atleast_1d(f[1](pre_data))
                else:
                    pre_data = np.squeeze(reco_vals[f[0]])
                    batch_input[:,j]=f[1](pre_data)
            inp_data.append(batch_input)
        
        # Generate Output Data
        if not use_data:
            for k, b in enumerate(out_branches):
                shape = (arr_size,)+out_branches[k][1]
                batch_output = np.zeros(shape)
                for j, f in enumerate(out_variables[k]):
                    pre_data = np.squeeze(reco_vals[f[0]])
                    if len(out_variables[k]) == 1:
                        batch_output[:]=np.reshape(f[1](pre_data), shape)
                    else:
                        batch_output[:,j] = f[1](pre_data)
                out_data.append(batch_output)

        #Prepare Next Loop
        ind_lo += batch_size
        ind_hi += batch_size
        if (ind_lo >= inds[cur_file][1]) | (equal_len & (ind_hi > inds[cur_file][1])):
            cur_file += 1
            if (cur_file == len(file_handlers)):
                cur_file=0
                new_inds = np.random.permutation(len(file_handlers))
                print('Shuffle filelist...')
                file_handlers = file_handlers[new_inds]
                inds = inds[new_inds] 
            t1 = time.time()
            print('\n Open File: {} \n'.format(file_handlers[cur_file]))
            print('\n Average Time per Batch: {}s \n'.format((t1-t0)/num_batches))
            t0 = time.time()
            num_batches = 1
            in_data.close()
            in_data = h5py.File(file_handlers[cur_file], 'r')
            f_reco_vals = in_data['reco_vals']
            ind_lo = inds[cur_file][0]
            ind_hi = ind_lo + batch_size
        elif ind_hi > inds[cur_file][1]:
            ind_hi = inds[cur_file][1]
       
        # Yield Result
        num_batches += 1
        if use_data:
            yield inp_data
        else:
            yield (inp_data, out_data, weights)


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
