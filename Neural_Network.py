#!/usr/bin/env python
# coding: utf-8


###### DeepIceLearning, a Project by Theo Glauch and Johannes Kager, 2017 #################

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
import sys
from configparser import ConfigParser
import socket

print('Running on Hostcomputer {}'.format(socket.gethostname()))

parser = ConfigParser()
try:
	parser.read('config.cfg')
except:
	raise Exception('Config File is missing!!!!')  

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
import argparse
import math
import time
import shelve
import shutil
# if backend == 'tensorflow':
#   from keras_exp.multigpu import get_available_gpus
#   from keras_exp.multigpu import make_parallel
from functions import *   

################# Function Definitions ########################################################

def parseArguments():

  """Parse the command line arguments

  Returns: 
  args : Dictionary containing the command line arguments

  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--project", help="The name for the Project", type=str ,default='some_NN')
  parser.add_argument("--input", help="Name of the input files seperated by :", type=str ,default='all')
  parser.add_argument("--model", help="Name of the File containing th qe model", type=str, default='simple_CNN.cfg')
  parser.add_argument("--virtual_len", help="Use an artifical array length (for debugging only!)", type=int , default=-1)
  parser.add_argument("--continue", help="Give a folder to continue the training of the network", type=str, default = 'None')
  parser.add_argument("--date", help="Give current date to identify safe folder", type=str, default = 'None')
  parser.add_argument("--ngpus", help="Number of GPUs for parallel training", type=int, default = 1)
  parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')
  args = parser.parse_args()
  return args

if __name__ == "__main__":

#################### Process Command Line Arguments ###########################################

  file_location = parser.get('Basics', 'thisfolder')
  mc_location = parser.get('Basics', 'mc_path')

  args = parseArguments()
  print("\n ---------")
  print("You are running the script with arguments: ")
  for a in args.__dict__:
      print('{} : {}'.format(a, args.__dict__[a]))
  print("--------- \n")

#################### Setup the Training Objects and Variables #################################

####### Continuing the training of a model ##############################

  if args.__dict__['continue'] != 'None':
    shelf = shelve.open(os.path.join(file_location, 
      args.__dict__['continue'], 
      'run_info.shlf'))

    project_name = shelf['Project']
    input_files = shelf['Files']
    train_inds = shelf['Train_Inds'] 
    valid_inds = shelf['Valid_Inds']
    test_inds = shelf['Test_Inds']
    model = load_model(os.path.join(file_location, args.__dict__['continue'], 'best_val_loss.npy'))
    today = args.__dict__['continue'].split('/')[1]
    print(today)
    shelf.close()

####### Build-up a new Model ###########################################

  else:
    project_name = args.__dict__['project']

    if args.__dict__['input'] =='all':
      input_files = [file for file in \
      os.listdir(mc_location) \
      if os.path.isfile(os.path.join(mc_location, file))]
    else:
      input_files = (args.__dict__['input']).split(':')

    ## Create Folders
    if args.__dict__['date'] != 'None':
      today = args.__dict__['date']
    else:
      today = datetime.date.today()
    if 'save_path' in parser['Basics'].keys():
      save_path =  parser.get('Basics', 'save_path')
    else:
      save_path = os.path.join(file_location,'train_hist/{}/{}'.format(today, project_name))

    if not os.path.exists(save_path):
      os.makedirs(save_path)
    
    train_val_test_ratio=[float(parser.get('Training_Parameters', 'training_fraction')),
    float(parser.get('Training_Parameters', 'validation_fraction')),
    float(parser.get('Training_Parameters', 'test_fraction'))] 

    file_len = read_input_len_shapes(mc_location, 
      input_files, 
      virtual_len = args.__dict__['virtual_len'])

    train_frac  = float(train_val_test_ratio[0])/np.sum(train_val_test_ratio)
    valid_frac = float(train_val_test_ratio[1])/np.sum(train_val_test_ratio)
    train_inds = [(0, int(tot_len*train_frac)) for tot_len in file_len] 
    valid_inds = [(int(tot_len*train_frac), int(tot_len*(train_frac+valid_frac))) for tot_len in file_len] 
    test_inds = [(int(tot_len*(train_frac+valid_frac)), tot_len-1) for tot_len in file_len] 
    print('Index ranges used for training: {}'.format(train_inds))

    ### Create the Model
    conf_model_file = os.path.join('Networks', args.__dict__['model'])
    model_settings, model_def = parse_config_file(conf_model_file)
    shapes, shape_names, inp_variables, inp_transformations, out_variables, out_transformations = \
     prepare_input_output_variables(os.path.join(mc_location, input_files[0]), model_settings)

    ngpus = args.__dict__['ngpus']
    adam = keras.optimizers.Adam(lr=float(parser.get('Training_Parameters', 'learning_rate')))

    if ngpus > 1 :
      if backend == 'tensorflow':
        with tf.device('/cpu:0'):
          # define the serial model.
          model_serial = base_model(model_def, shapes, shape_names)

        gdev_list = get_available_gpus()
        print('Using GPUs: {}'.format(gdev_list))
        model = make_parallel(model_serial, gdev_list)
      else:
        raise Exception('Multi GPU can only be used with tensorflow as Backend.')
    else:
      model = base_model(model_def, shapes, shape_names)

    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    os.system("nvidia-smi")  

    ## Save Run Information
    shelf = shelve.open(os.path.join(save_path,'run_info.shlf'))
    shelf['Project'] = project_name
    shelf['Files'] = args.__dict__['input']
    shelf['Train_Inds'] = train_inds
    shelf['Valid_Inds'] = valid_inds
    shelf['Test_Inds'] = test_inds
    shelf.close()

    shutil.copy(conf_model_file, os.path.join(file_location,
      'train_hist/{}/{}/model.cfg'.format(today, project_name)))

#################### Train the Model #########################################################

  CSV_log = keras.callbacks.CSVLogger( \
    os.path.join(save_path, 'loss_logger.csv'), 
    append=True)
  
  early_stop = keras.callbacks.EarlyStopping(\
    monitor='val_loss',
    min_delta = int(parser.get('Training_Parameters', 'delta')), 
    patience = int(parser.get('Training_Parameters', 'patience')), 
    verbose = int(parser.get('Training_Parameters', 'verbose')), 
    mode = 'auto')

  best_model = keras.callbacks.ModelCheckpoint(\
    os.path.join(save_path,'best_val_loss.npy'), 
    monitor = 'val_loss', 
    verbose = int(parser.get('Training_Parameters', 'verbose')), 
    save_best_only = True, 
    mode='auto', 
    period=1)

  batch_size = ngpus*int(parser.get('Training_Parameters', 'single_gpu_batch_size'))

  model.fit_generator(generator(batch_size, mc_location, input_files, train_inds, shapes, inp_variables, inp_transformations, out_variables, out_transformations), 
                steps_per_epoch = math.ceil(np.sum([k[1]-k[0] for k in train_inds])/batch_size),
                validation_data = generator(batch_size, mc_location, input_files, valid_inds, shapes, inp_variables, inp_transformations, out_variables, out_transformations),
                validation_steps = math.ceil(np.sum([k[1]-k[0] for k in valid_inds])/batch_size),
                callbacks = [CSV_log, early_stop, best_model, MemoryCallback()], 
                epochs = int(parser.get('Training_Parameters', 'epochs')), 
                verbose = int(parser.get('Training_Parameters', 'verbose')),
                max_q_size = int(parser.get('Training_Parameters', 'max_queue_size')),
                )

#################### Saving the Final Model and Calculation/Saving of Result for Test Dataset ######################

  print('\n Save the Model \n')
  model.save(os.path.join(save_path,'final_network.h5'))  # save trained network

  print('\n Calculate Results... \n')
  prediction = model.predict_generator(generator(batch_size, mc_location, input_files, test_inds, shapes, inp_variables, inp_transformations, out_variables, out_transformations), 
                steps = math.ceil(np.sum([k[1]-k[0] for k in test_inds])/batch_size),
                verbose = int(parser.get('Training_Parameters', 'verbose')),
                max_q_size = int(parser.get('Training_Parameters', 'max_queue_size'))
                )

  MC_truth = []
  for i in range(len(input_data)):
    one_chunk = np.log10(output_data[i][test_inds[i][0]:test_inds[i][1],0:1])
    MC_truth.extend(list(one_chunk))


  np.save(os.path.join(file_location,'train_hist/{}/{}/test_results.npy'.format(today, project_name)), 
    [prediction, np.squeeze(MC_truth)])

  print(' \n Finished .... Exit.....')
