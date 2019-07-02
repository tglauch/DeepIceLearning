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

import sys
import os
from six.moves import configparser
import socket
import argparse
import h5py
sys.path.append(os.path.join(os.path.abspath(".."),'lib'))
print os.path.abspath("..")
import model_parse as mp
import importlib


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
    return args.__dict__


# Read config and load keras stuff #############

print('Running on Hostcomputer {}'.format(socket.gethostname()))
args = parseArguments()
parser = configparser.ConfigParser()
if args['continue'] != 'None' and args['main_config'] == 'None':
    save_path = args['continue']
    config_file = os.path.join(save_path, 'config.cfg')
else:
    config_file = args['main_config']
try:
    parser.read(config_file)
except Exception:
    raise Exception('Config File is missing!!!!')

parser_dict = {s: dict(parser.items(s)) for s in parser.sections()}
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
print(os.environ['LD_LIBRARY_PATH'])
print(os.environ['PATH'])
if backend == 'tensorflow':
    print('Run with backend Tensorflow')
    import tensorflow as tf
    print('Version {}, \n Path {}'.format(tf.__version__, tf.__path__))
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
from keras.utils import multi_gpu_model
from keras.utils import plot_model
from keras.callbacks import CSVLogger, EarlyStopping
#import individual_loss
import transformations
from functions import *

if __name__ == "__main__":

    # Process Command Line Arguments 

    print("\n ---------")
    print("You are running the script with arguments: ")
    for a in args.keys():
        print('{} : {}'.format(a, args[a]))
    print("--------- \n")

    if args['continue'] != 'None':
        save_path = args['continue']
        run_info = np.load(os.path.join(save_path, 'run_info.npy'))[()]

        mc_location = parser.get('Basics', 'mc_path')
        input_files = run_info['Files']
        if input_files == "['all']":
            input_files = os.listdir(mc_location)
        conf_model_file = args['model']
        print "Continuing training. Loaded dict : ", run_info
        print "Input files: ", input_files

    # Build-up a new Model

    else:
        mc_location = parser.get('Basics', 'mc_path')
        conf_model_file = args['model']
        if args['input'] == 'all':
            input_files = [f for f in os.listdir(mc_location)
                           if os.path.isfile(
                           os.path.join(mc_location, f)) and f[-3:] == '.h5']
            print('Use the following input files for training: {}'.
                  format(input_files))
        else:
            input_files = (args['input']).split(':')

        if args['save_folder'] != 'None':
            save_path = args['save_folder']
        elif 'save_path' in parser_dict['Basics'].keys():
            save_path = parser.get('Basics', 'save_path')
        elif 'train_folder' in parser_dict["Basics"].keys():
            today = str(datetime.datetime.now()).\
                replace(" ", "-").split(".")[0].replace(":", "-")
            project_name = args['project']
            save_path = os.path.join(
                parser.get('Basics', 'train_folder'),
                '{}/{}'.format(project_name, today))
        else:
            raise Exception(
                'I have no clue where to save the training results')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path + "/model_all_epochs"):
        os.makedirs(save_path + "/model_all_epochs")
    if not os.path.exists(save_path + "/model_all_epochs/batch"):
        os.makedirs(save_path + "/model_all_epochs/batch")


    train_val_test_ratio = [
        float(parser.get('Training_Parameters', 'training_fraction')),
        float(parser.get('Training_Parameters', 'validation_fraction')),
        float(parser.get('Training_Parameters', 'test_fraction'))]

    file_len = read_input_len_shapes(mc_location,
                                     input_files,
                                     virtual_len=args['virtual_len'])
    train_frac = float(
        train_val_test_ratio[0]) / np.sum(train_val_test_ratio)
    valid_frac = float(
        train_val_test_ratio[1]) / np.sum(train_val_test_ratio)
    train_inds = [(0, int(tot_len * train_frac)) for tot_len in file_len]
    valid_inds = [(int(tot_len * train_frac),
                  int(tot_len * (train_frac + valid_frac)))
                  for tot_len in file_len]
    test_inds = [(int(tot_len * (train_frac + valid_frac)), tot_len)
                 for tot_len in file_len]
    print('Index ranges used for training: {} \n'.format(train_inds))
    print('Index ranges used for validation: {} \n'.format(valid_inds))
    print('Index ranges used for testing: {} \n'.format(test_inds))

    w_func_str = parser.get('Training_Parameters','weighting')
    print('Use Weighting Function {}'.format(w_func_str))
    if w_func_str != 'None':
        mod = importlib.import_module('weighting')
        w_func = getattr(mod, w_func_str)
        w_func_gen = w_func(input_files, mc_location)
    else:
        w_func_gen = None 

    # create model (new implementation, functional API of Keras)
    base_model, inp_shapes, inp_trans, out_shapes, out_trans, loss_dict, mask_func = \
        mp.parse_functional_model(
            conf_model_file,
            os.path.join(mc_location, input_files[0]))

    # Choosing the Optimizer
    optimizer_used = chose_optimizer(parser.get('Training_Parameters', 'optimizer'),
                                    float(parser.get('Training_Parameters', 'learning_rate')))

    # Multi GPU stuff
    ngpus = args['ngpus']
    if ngpus > 1:
        model_serial = read_NN_weights(args, base_model)
        model = multi_gpu_model(model_serial, gpus=ngpus)
        equal_len = True
    else:
        model = read_NN_weights(args, base_model)
        model_serial = model
        equal_len = False

    # Compile the model with the given settings
    model.compile(optimizer=optimizer_used, **loss_dict)
    print(os.system("nvidia-smi"))


    # save run info
    if args['continue'] == 'None':
        run_info = dict()
        run_info['Files'] = input_files
        run_info['mc_location'] = mc_location
        run_info['Test_Inds'] = test_inds
        run_info['inp_shapes'] = inp_shapes
        run_info['out_shapes'] = out_shapes
        run_info['inp_trans'] = inp_trans
        run_info['out_trans'] = out_trans
        #run_info['loss_dict'] = loss_dict
      
        np.save(os.path.join(save_path, 'run_info.npy'), run_info)

# Train the Model 
    batch_size = int(parser.get("GPU", "request_gpus")) * int(
                     parser.get('Training_Parameters', 'single_gpu_batch_size'))
    file_handlers = [os.path.join(mc_location, file_name)
                     for file_name in input_files]

    # saving model every epoch
    all_epoch_folder = os.path.join(save_path, "model_all_epochs")
    if not os.path.exists(all_epoch_folder):
        os.makedirs(all_epoch_folder)
        os.makedirs(os.path.join(all_epochs_folder, "batch"))
    print('Created Folder {}'.format(all_epoch_folder))

    divider = int(parser.get('Training_Parameters', 'epoch_divider'))
    training_steps = int(np.sum([math.ceil((1.*(k[1]-k[0])/batch_size)) for k in train_inds])/divider)
    validation_steps = int(np.sum([math.ceil((1.*(k[1]-k[0])/batch_size)) for k in valid_inds]))
    
    best_model = ParallelModelCheckpoint(
        model = model_serial,
        filepath= os.path.join(save_path, "best_val_loss.npy"),
        monitor='val_loss',
        verbose=int(parser.get('Training_Parameters', 'verbose')),
        save_best_only=True,
        mode='auto',
        period=1)

    model.fit_generator(
        generator_v2(
            batch_size, file_handlers, train_inds, inp_shapes, inp_trans,
            out_shapes, out_trans, weighting_function=w_func_gen,
            equal_len=equal_len, mask_func=mask_func),
        steps_per_epoch=training_steps,
        validation_data=generator_v2(
            batch_size, file_handlers, valid_inds, inp_shapes,
            inp_trans, out_shapes, out_trans, weighting_function=w_func_gen,
            equal_len=equal_len, valid=True),
        validation_steps=validation_steps,
        callbacks=[CSVLogger(os.path.join(save_path,'loss_logger.csv'), append=True),
                   EarlyStopping(min_delta=int(parser.get('Training_Parameters', 'delta')),
                                 patience=int(parser.get('Training_Parameters', 'patience')),
                                 verbose=int(parser.get('Training_Parameters', 'verbose')),
                                 monitor='val_loss'),
                    best_model,
#                   every_model(model_serial,
#                              os.path.join(save_path, "model_all_epochs/weights_{epoch:02d}.npy"),
#                              int(parser.get('Training_Parameters', 'verbose'))),
                   MemoryCallback()],
                #   WeightsSaver(int(parser.get('Training_Parameters', 'save_every_x_batches')), save_path)],
        epochs=int(parser.get('Training_Parameters', 'epochs')),
        verbose=int(parser.get('Training_Parameters', 'verbose')),
        max_queue_size=int(parser.get('Training_Parameters', 'max_queue_size')),
        use_multiprocessing=False)

    print('\n Finished .... Exit.....')

