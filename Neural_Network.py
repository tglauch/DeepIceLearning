#! /usr/bin/env python
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

import os
from six.moves import configparser
#changed this because ConfigParser was not available on the RZ in Aachen
import socket
import argparse
import h5py
import tables
import lib.model_parse as mp
import sys
import numpy.lib.recfunctions as rfn

# Function Definitions #####################
def parseArguments():
    """Parse the command line arguments

    Returns:
    args : Dictionary containing the command line arguments

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main_config",
        help="main config file, user-specific",
        type=str, default='default.cfg')
    parser.add_argument(
        "--project", help="The name for the Project",
        type=str, default='some_NN')
    parser.add_argument(
        "--input",
        help="Name of the input files seperated by :",
        type=str, default='all')
    parser.add_argument(
        "--model",
        help="Name of the File containing the model",
        type=str, default='simple_CNN.cfg')
    parser.add_argument(
        "--virtual_len",
        help="Use an artifical array length (for debugging only!)",
        type=int, default=-1)
    parser.add_argument(
        "--continue",
        help="Absolute path to a folder to continue training of the network",
        type=str, default='None')
    parser.add_argument(
        "--load_weights",
        help="Give a path to pre-trained model weights",
        type=str, default='None')
    parser.add_argument(
        "--ngpus",
        help="Number of GPUs for parallel training",
        type=int, default=1)
    parser.add_argument(
        "--version",
        action="version", version='%(prog)s - Version 1.0')
    parser.add_argument(
        "--save_folder",
        help="Folder for saving the output",
        type=str, default='None')
    args = parser.parse_args()
    return args


# Read config and load keras stuff #############

print('Running on Hostcomputer {}'.format(socket.gethostname()))
args = parseArguments()
parser = configparser.ConfigParser()
if args.__dict__['continue'] != 'None' and args.main_config=='None':
    save_path = args.__dict__['continue']
    config_file = os.path.join(save_path, 'config.cfg')
else:
    config_file = args.main_config
try:
    parser.read(config_file)
except Exception:
    raise Exception('Config File is missing!!!!')

parser_dict = {s:dict(parser.items(s)) for s in parser.sections()}
backend = parser.get('Basics', 'keras_backend')

os.environ["KERAS_BACKEND"] = backend
if backend == 'theano':
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

cuda_path = parser.get('Basics', 'cuda_installation')
if not os.path.exists(cuda_path):
    raise Exception('Given Cuda installation does not exist!')

if cuda_path not in os.environ['LD_LIBRARY_PATH'].split(os.pathsep):
    print('Setting Cuda Path...')
    os.environ["PATH"] += os.pathsep + cuda_path
    os.environ['LD_LIBRARY_PATH'] += os.pathsep + cuda_path
    try:
        print('Attempt to Restart with new Cuda Path')
        os.execv(sys.argv[0], sys.argv)
    except Exception, exc:
        print 'Failed re-exec:', exc
    sys.exit(1)

if backend == 'tensorflow':
    print('Run with backend Tensorflow')
    import tensorflow as tf
elif backend == 'theano':
    print('Run with backend Theano')
    import theano
else:
    raise NameError('Choose tensorflow or theano as keras backend')


import numpy as np
import datetime
import math
import argparse
import time
import shelve
if backend == 'tensorflow':
    from keras_exp.multigpu import get_available_gpus
    from keras_exp.multigpu import make_parallel
from lib.functions import *
from keras.utils import plot_model
import lib.individual_loss

if __name__ == "__main__":

    # Process Command Line Arguments #########################################

    file_location = parser.get('Basics', 'thisfolder')

    print("\n ---------")
    print("You are running the script with arguments: ")
    for a in args.__dict__:
        print('{} : {}'.format(a, args.__dict__[a]))
    print("--------- \n")

    # Setup the Training Objects and Variables ##############################

    # Continuing the training of a model ##############################
    # Has to be changed because shelf was substituded
    #if args.__dict__['continue'] != 'None':
        #save_path = args.__dict__['continue']
        #shelf = shelve.open(os.path.join(save_path, 'run_info.shlf'))
        #mc_location = shelf['mc_location']
        #input_files = shelf['Files']
        #if input_files == "['all']":
            #input_files = os.listdir(mc_location)
        #conf_model_file = os.path.join(save_path, 'model.py')
        #print "Continuing training. Loaded shelf : ", shelf
        #print "Input files: ", input_files
        #shelf.close()

    # ALTERNATIVE TO SHELF, using a simple dict
    if args.__dict__['continue'] != 'None':
        save_path = args.__dict__['continue']
        run_info =  np.load(os.path.join(save_path, 'run_info.npy'))[()]
        #mc_location = run_info['mc_location']
        #################################################################
        mc_location = parser.get('Basics', 'mc_path')
        ###############################################################
        input_files = run_info['Files']
        if input_files == "['all']":
            input_files = os.listdir(mc_location)
        conf_model_file = args.__dict__['model']
        print "Continuing training. Loaded dict : ", run_info
        print "Input files: ", input_files

    # Build-up a new Model ###########################################

    else:
        mc_location = parser.get('Basics', 'mc_path')
        conf_model_file = args.__dict__['model']
        if args.__dict__['input'] == 'all':
            input_files = [f for f in os.listdir(mc_location)
                           if os.path.isfile(
                           os.path.join(mc_location, f)) and f[-3:] == '.h5']
            print('Use the following input files for training: {}'.
                  format(input_files))
        else:
            input_files = (args.__dict__['input']).split(':')

        if args.__dict__['save_folder'] != 'None':
            save_path = args.__dict__['save_folder']
        elif 'save_path' in parser_dict['Basics'].keys():
            save_path = parser.get('Basics', 'save_path')
        elif 'train_folder' in parser_dict["Basics"].keys():
            today = str(datetime.datetime.now()).\
                replace(" ", "-").split(".")[0].replace(":", "-")
            project_name = args.__dict__['project']
            save_path = os.path.join(
                parser.get('Basics', 'train_folder'),
                '{}/{}'.format(project_name, today))
        else:
            raise Exception(
                'I have no clue where to save the training results')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_val_test_ratio = [
        float(parser.get('Training_Parameters', 'training_fraction')),
        float(parser.get('Training_Parameters', 'validation_fraction')),
        float(parser.get('Training_Parameters', 'test_fraction'))]

    file_len = read_input_len_shapes(mc_location,
                                     input_files,
                                     virtual_len=args.__dict__['virtual_len'])
    train_frac = float(
        train_val_test_ratio[0]) / np.sum(train_val_test_ratio)
    valid_frac = float(
        train_val_test_ratio[1]) / np.sum(train_val_test_ratio)
    train_inds = [(0, int(tot_len * train_frac)) for tot_len in file_len]
    valid_inds = [(int(tot_len * train_frac),
                  int(tot_len * (train_frac + valid_frac)))
                  for tot_len in file_len]
    test_inds = [(int(tot_len * (train_frac + valid_frac)), tot_len - 1)
                 for tot_len in file_len]
    print('Index ranges used for training: {}'.format(train_inds))
    print('Index ranges used for validation: {}'.format(valid_inds))
    print('Index ranges used for testing: {}'.format(test_inds))

    # create model (new implementation, functional API of Keras)
    base_model, inp_shapes, inp_trans, out_shapes, out_trans = \
        mp.parse_functional_model(
            conf_model_file,
            os.path.join(mc_location, input_files[0]))


    # Choosing the Optimizer
    if parser.get('Training_Parameters', 'optimizer')=="Nadam":
        print "Optimizer: Nadam"
        optimizer_used = keras.optimizers.Nadam(
             lr=float(parser.get('Training_Parameters', 'learning_rate')))
    elif parser.get('Training_Parameters', 'optimizer')=="Adam":
        optimizer_used = keras.optimizers.Adam(
             lr=float(parser.get('Training_Parameters', 'learning_rate')))
    else:
        print "Optimizer unchoosen or unknown -> default: Adam"
        optimizer_used = keras.optimizers.Adam(
             lr=float(parser.get('Training_Parameters', 'learning_rate')))

    ngpus = args.__dict__['ngpus']
    #print'Use {} GPUS'.format(ngpus)
    if ngpus > 1:
        if backend == 'tensorflow':
            with tf.device('/cpu:0'):
                model_serial = read_NN_weights(args.__dict__, base_model)
            gdev_list = get_available_gpus()
            print('Using GPUs: {}'.format(gdev_list))
            model = make_parallel(model_serial, gdev_list)
        else:
            raise Exception(
                'Multi GPU can only be used with tensorflow as Backend.')
    else:
        model = read_NN_weights(args.__dict__, base_model)

    # Choosing the Loss function
    loss_func = 'mean_squared_error'
    if parser.has_option('Training_Parameters', 'loss_function'):
        loss_func = parser.get('Training_Parameters', 'loss_function')
        if loss_func == "weighted_categorial_crossentropy":
            weights = parser.get('Training_Parameters', 'weights')
            weights = np.array(weights.split(',')).astype(np.float)
            loss_func = lib.individual_loss.weighted_categorical_crossentropy(weights) 
    print "Used Loss-Function {}".format(loss_func)
###########################################################################################
    if parser.has_option('Multi_Task_Learning', 'ON/OFF') == "ON":
        if parser.has_option('Multi_Task_Learning', 'CustomLoss') == "ON":
            #extra_parameter_1 = XX
            #extra_parameter_2 =
            #custom = [individual_loss.event_type_and_energy_weighted_loss(extra_parameter_1, extra_parameter_2), "categorial_crossentropy", "categorial_crossentropy", "categorial_crossentropy"] 
            
##############################################################################################
            weights = parser.get('Multi_Task_Learning', 'weights')
            weights = np.array(weights.split(',')).astype(np.float)
            custom = [lib.individual_loss.weighted_categorical_crossentropy(weights), "categorial_crossentropy", "categorial_crossentropy", "categorial_crossentropy"]
            

############################################################################################################
            model.compile(optimizer=optimizer_used,\
                loss=custom,
                loss_weights = parser.has_option('Multi_Task_Learning', 'loss_weights'))
        else:   
            model.compile(optimizer=optimizer_used,\
               loss = parser.has_option('Multi_Task_Learning', 'loss'),\
               loss_weights = parser.has_option('Multi_Task_Learning', 'loss_weights'))
    else: 
        if loss_func == "weighted_categorial_crossentropy":
            model.compile(
                loss=loss_func, optimizer=optimizer_used)
        else:
            model.compile(
                loss=loss_func, optimizer=optimizer_used, metrics=['accuracy'])
###########################################################################################
    os.system("nvidia-smi")

    # Save Run Information
    if not os.path.exists(os.path.join(save_path, 'run_info.shlf')):
        if args.__dict__['continue'] == 'None':
            shelf = shelve.open(os.path.join(save_path, 'run_info.shlf'))
            shelf['Files'] = input_files
            shelf['mc_location'] = mc_location
            shelf['Test_Inds'] = test_inds
            shelf.close()

    # Alternative to Shelf
    if not os.path.exists(os.path.join(save_path, "run_info.npy")):
        if args.__dict__['continue'] == 'None':
            run_info = dict()
            run_info['Files'] = input_files
            run_info['mc_location'] = mc_location
            run_info['Test_Inds'] = test_inds
            np.save(os.path.join(save_path, 'run_info.npy'), run_info)

# Train the Model #########################################################

    CSV_log = keras.callbacks.CSVLogger(
        os.path.join(save_path,
                     'loss_logger.csv'),
        append=True)

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=int(parser.get('Training_Parameters', 'delta')),
        patience=int(parser.get('Training_Parameters', 'patience')),
        verbose=int(parser.get('Training_Parameters', 'verbose')),
        mode='auto')

    best_model = keras.callbacks.ModelCheckpoint(
        save_path + "/best_val_loss.npy",
        monitor='val_loss',
        verbose=int(parser.get('Training_Parameters', 'verbose')),
        save_best_only=True,
        mode='auto',
        period=1)

    batch_size = int(
        parser.get("GPU", "request_gpus")) * int(parser.get(
            'Training_Parameters', 'single_gpu_batch_size'))
    file_handlers = [h5py.File(os.path.join(mc_location, file_name))
                     for file_name in input_files]

    #file_handlers = [tables.open_file(os.path.join(mc_location, file_name))
    #                 for file_name in input_files]

    epoch_divider = int(parser.get('Training_Parameters', 'epoch_divider'))
    model.fit_generator(
        generator(
            batch_size, file_handlers, train_inds, inp_shapes,
            inp_trans, out_shapes, out_trans),
        steps_per_epoch=math.ceil(\
            #(np.sum([k[1] - k[0] for k in train_inds]) / batch_size))/len(input_files),
            (np.sum([k[1] - k[0] for k in train_inds]) / batch_size))/epoch_divider,
        validation_data=generator(\
            batch_size, file_handlers, valid_inds, inp_shapes,
            inp_trans, out_shapes, out_trans, val_run=True),
        validation_steps=math.ceil(\
            np.sum([k[1] - k[0] for k in valid_inds]) / batch_size)/epoch_divider,
        callbacks=[CSV_log, early_stop, best_model, MemoryCallback()],
        epochs=int(parser.get('Training_Parameters', 'epochs')),
        verbose=int(parser.get('Training_Parameters', 'verbose')),
        max_q_size=int(parser.get('Training_Parameters', 'max_queue_size')))


    # Saving a visualization of the model 
    plot_model(model, to_file=os.path.join(save_path, 'model.pdf'))
    print('\n Model Visualisation saved')


    # Saving the Final Model and Calculation/Saving of Result for Test Dataset ####

    model.save(os.path.join(save_path, 'final_network.h5'))  # save trained network
    print('\n Saved the Model \n')

    print('\n Finished .... Exit.....')

