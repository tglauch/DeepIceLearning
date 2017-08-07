#!/usr/bin/env python
# coding: utf-8

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
cuda_path = parser.get('Basics', 'cuda_installation')
os.environ["PATH"] += os.pathsep + cuda_path

if not os.path.exists(cuda_path):
	raise Exception('Given Cuda installation does not exist!')

if backend == 'tensorflow':
	print('Run with backend Tensorflow')
	import tensorflow as tf
elif backend == 'theano':
	print('Run with backend Theano')
	import theano
	os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32" 
else:
	raise NameError('Choose tensorflow or theano as keras backend')

import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D,\
 BatchNormalization, MaxPooling2D,Convolution3D,MaxPooling3D, Merge
from keras import regularizers
import h5py
import datetime
import argparse
import math
import time
import resource
import shelve
import shutil
from keras_exp.multigpu import get_available_gpus
from keras_exp.multigpu import make_parallel

################# Function Definitions ####################################################################

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
  ### not used atm...normalization has to be defined in the config file
  parser.add_argument('--normalize_input', dest='norm_input', action='store_true')
  parser.set_defaults(norm_input=True)
  parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')
  args = parser.parse_args()
  return args

def read_files(input_files, virtual_len=-1):

  """Create an Array of Input and Output HDF5 Monte Carlo data from a given list of files(e.g file1:file2 ...) 

  Arguments:
  input_shape : The file list as above
  virtual_len : can be set for debugging purposes if only the first $virtual_len events shell be considered 

  Returns: 
  input_data : A list of datafiles to be feeded into the network
  output_data : A list of datafiles as true value for the network
  file_len : The number of events for each file

  """

  input_data = []
  output_data = []
  file_len = []
  inp_shape = []
  print input_files
  for run, input_file in enumerate(input_files):
    data_file = os.path.join(file_location, 'training_data/{}'.format(input_file))

    if virtual_len == -1:
      data_len = len(h5py.File(data_file)['charge'])
    else:
      data_len = virtual_len
      print('Only use the first {} Monte Carlo Events'.format(data_len))

    this_input = h5py.File(data_file, 'r')['charge']
    input_data.append(this_input)
    this_shape = np.shape(this_input[0])
    if run == 0:
      inp_shape=this_shape 
    else:
      if not this_shape == inp_shape:
        raise Exception('The input shape of the data contained in the input files does not match')

    output_data.append(h5py.File(data_file, 'r')['reco_vals'])
    file_len.append(data_len)

  return input_data, output_data, file_len

def prepare_input_shapes(one_input_array, model_settings):
  shapes = []
  shape_names = []
  if len(model_settings) == 0:
    model_settings = ['[Inputs]', 'model = x']
  for block in model_settings:
    if block[0]=='[Inputs]':
      for i in range(1,len(block)):
        this_definition = block[i].split('=')
        shape_names.append(this_definition[0].strip())
        pre_shape = np.shape(eval(this_definition[1].strip().replace('x', 'one_input_array')))
        if pre_shape == ():
          shapes.append((1,))
        else:
          shapes.append(pre_shape)
  return shapes, shape_names

def parse_config_file(conf_file_path):

  f = open(conf_file_path)
  config_array = f.read().splitlines()
  config_blocks = []
  single_block = []
  for line in config_array:
      if line=='':
          config_blocks.append(single_block)
          single_block = []
      else:
          single_block.append(line)
  settings =[]
  model = []
  mode = ''
  for block in config_blocks:
      if mode =='' or block[0][0]=='*':
          if block[0] == '*Settings*':
              mode = 'settings'
          elif block[0] == '*Model*':
              mode = 'model'
          else:
              raise Exception('config file is corrupted')
      else:
          if mode=='settings':
              settings.append(block)
          elif mode=='model':
              model.append(block)

  return settings, model

def add_layer(model, layer, args, kwargs):

  """Given the data read from the network configuration file, add a layer to the Keras xnetwork model object

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
  model_def : (Relative) Path to the config (definition) file of the neural network

  Returns: 
  model : the (non-compiled) model object
  inp_shape : the required shape of the input data

  """
  models = dict()
  cur_model = None
  cur_model_name = ''
  print shapes
  print shape_names
  for block in model_def:
      if block[0][0] == '{' and block[0][-1] == '}' or cur_model == None:
          if cur_model != None:
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
          layer=block[0][1:-1]
          for i in range(1,len(block)):
              if block[i]=='[kwargs]':
                  mode = 'kwargs'
              elif mode == 'args':
                  try:
                      args.append(eval(block[i].split('=')[1].strip()))
                  except:
                      args.append(block[i].split('=')[1].strip())
              elif mode == 'kwargs':
                  split_line = block[i].split('=')
                  try:
                      kwargs[split_line[0].strip()] = eval(split_line[1].strip())
                  except:
                      kwargs[split_line[0].strip()] = split_line[1].strip()   
          if not layer == 'Merge':
              if not 'input_shape' in kwargs and input_layer==True:
                ind = shape_names.index(cur_model_name)
                kwargs['input_shape']=shapes[ind]
              print kwargs
              add_layer(cur_model, layer, args,kwargs)
          else:
              merge_layer_names = [name.strip() for name in kwargs['layers'][1:-1].split(',')]
              kwargs = dict()
              kwargs['mode']='concat'
              add_layer(cur_model, layer,[[models[name] for name in merge_layer_names]], kwargs)
              for name in merge_layer_names:
                  del models[name] 
          input_layer = False
  print(cur_model.summary())
  models[cur_model_name] = cur_model  
  return cur_model
  
  # print(model.summary())
  # return model, eval(str(inp_shape))

class MemoryCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, log={}):
        print(' \n RAM Usage {:.2f} GB \n \n'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))
        os.system("nvidia-smi")

def generator(batch_size, input_data, output_data, inds, inp_shape, model_settings):

  """Generator to create the mini-batches feeded to the network.

  Arguments:
  model : (Relative) Path to the config (definition) file of the neural network

  Returns: 
  model : the (non-compiled) model object
  inp_shape : the required shape of the input data

  """

  batch_input = [ np.zeros((batch_size,)+i) for i in inp_shape ]
  batch_out = np.zeros((batch_size,1))
  cur_file = 0
  cur_event_id = inds[cur_file][0]
  cur_len = 0
  up_to = inds[cur_file][1]
  for i in model_settings:
    if i[0] == '[Inputs]':
      input_transformation = i[1:]
  print(input_transformation)
  transformations = [i.split('=')[1].strip().replace('x', 'temp_in[i]') for i in input_transformation]
  print(transformations)
  while True:
    temp_in = []
    temp_out = []
    while cur_len<batch_size:
      fill_batch = batch_size-cur_len
      if fill_batch < (up_to-cur_event_id):
        temp_in.extend(input_data[cur_file][cur_event_id:cur_event_id+fill_batch])
        temp_out.extend(output_data[cur_file][cur_event_id:cur_event_id+fill_batch])
        cur_len += fill_batch
        cur_event_id += fill_batch
      else:
        temp_in.extend(input_data[cur_file][cur_event_id:up_to])
        temp_out.extend(output_data[cur_file][cur_event_id:up_to])
        cur_len += up_to-cur_event_id
        cur_file+=1
        if cur_file == len(inds):
          cur_file = 0
          cur_event_id = inds[cur_file][0]
          cur_len = 0
          up_to = inds[cur_file][1]
          break
        else:
          cur_event_id = inds[cur_file][0]
          up_to = inds[cur_file][1]
    for i in range(len(temp_in)):
      for j, transform in enumerate(input_transformation):
          batch_input[j][i] = eval(transformations[j])
      batch_out[i] = np.log10(temp_out[i][0])
    cur_len = 0 
    yield (batch_input, batch_out)


if __name__ == "__main__":

#################### Process Command Line Arguments ######################################

  file_location = parser.get('Basics', 'thisfolder')

  args = parseArguments()
  print("\n ---------")
  print("You are running the script with arguments: ")
  for a in args.__dict__:
      print('{} : {}'.format(a, args.__dict__[a]))
  print("--------- \n")

######################## Setup the training variables ########################################################

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

    input_data, output_data, file_len = read_files(input_files.split(':'))

  else:
    project_name = args.__dict__['project']

    if args.__dict__['input'] =='all':
      input_files = [file for file in os.listdir(os.path.join(file_location, 'training_data/')) if os.path.isfile(os.path.join(file_location, 'training_data/', file))]
    else:
      input_files = (args.__dict__['input']).split(':')

    tvt_ratio=[float(parser.get('Training_Parameters', 'training_fraction')),
    float(parser.get('Training_Parameters', 'validation_fraction')),
    float(parser.get('Training_Parameters', 'test_fraction'))] 

    ## Create Folders
    if args.__dict__['date'] != 'None':
      today = args.__dict__['date']
    else:
      today = datetime.date.today()

    folders=['train_hist/',
     'train_hist/{}'.format(today),
     'train_hist/{}/{}'.format(today, project_name)]

    for folder in folders:
        if not os.path.exists('{}'.format(os.path.join(file_location,folder))):
            os.makedirs('{}'.format(os.path.join(file_location,folder)))

    input_data, output_data, file_len = read_files(input_files,
     virtual_len = args.__dict__['virtual_len'])

    train_frac  = float(tvt_ratio[0])/np.sum(tvt_ratio)
    valid_frac = float(tvt_ratio[1])/np.sum(tvt_ratio)
    train_inds = [(0, int(tot_len*train_frac)) for tot_len in file_len] 
    valid_inds = [(int(tot_len*train_frac), int(tot_len*(train_frac+valid_frac))) for tot_len in file_len] 
    test_inds = [(int(tot_len*(train_frac+valid_frac)), tot_len-1) for tot_len in file_len] 

    ### Create the Model
    conf_model_file = os.path.join('Networks', args.__dict__['model'])
    model_settings, model_def = parse_config_file(conf_model_file)
    shapes, shape_names = prepare_input_shapes(input_data[0][0], model_settings)
    ngpus = args.__dict__['ngpus']

    adam = keras.optimizers.Adam(lr=float(parser.get('Training_Parameters', 'learning_rate')))
    if ngpus > 1 :
      with tf.device('/cpu:0'):
        # define the serial model.
        ##### TODO Include the correct input shape of the data as calculated from the input files.
        ## Why? Don't want to rely on user giving the correct input shape

        model_serial = base_model(model_def, shapes, shape_names)

      gdev_list = get_available_gpus()
      print('Using GPUs: {}'.format(gdev_list))
      model = make_parallel(model_serial, gdev_list)
    else:
      model = base_model(model_def, shapes, shape_names)

    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    os.system("nvidia-smi")  

    ## Save Run Information
    shelf = shelve.open(os.path.join(file_location,'train_hist/{}/{}/run_info.shlf'.format(today, project_name)))
    shelf['Project'] = project_name
    shelf['Files'] = args.__dict__['input']
    shelf['Train_Inds'] = train_inds
    shelf['Valid_Inds'] = valid_inds
    shelf['Test_Inds'] = test_inds
    shelf.close()

    shutil.copy(conf_model_file, os.path.join(file_location,'train_hist/{}/{}/model.cfg'.format(today, project_name)))

#################### Train the Model #########################################################

  CSV_log = keras.callbacks.CSVLogger( \
    os.path.join(file_location,'train_hist/{}/{}/loss_logger.csv'.format(today, project_name)), 
    append=True)
  
  early_stop = keras.callbacks.EarlyStopping(\
    monitor='val_loss',
    min_delta = int(parser.get('Training_Parameters', 'delta')), 
    patience = int(parser.get('Training_Parameters', 'patience')), 
    verbose = int(parser.get('Training_Parameters', 'verbose')), 
    mode = 'auto')

  best_model = keras.callbacks.ModelCheckpoint(\
    os.path.join(file_location,'train_hist/{}/{}/best_val_loss.npy'.format(today, project_name)), 
    monitor = 'val_loss', 
    verbose = int(parser.get('Training_Parameters', 'verbose')), 
    save_best_only = True, 
    mode='auto', 
    period=1)

  batch_size = ngpus*int(parser.get('Training_Parameters', 'single_gpu_batch_size'))
  model.fit_generator(generator(batch_size, input_data, output_data, train_inds, shapes, model_settings), 
                steps_per_epoch = math.ceil(np.sum([k[1]-k[0] for k in train_inds])/batch_size),
                validation_data = generator(batch_size, input_data, output_data, valid_inds, shapes, model_settings),
                validation_steps = math.ceil(np.sum([k[1]-k[0] for k in valid_inds])/batch_size),
                callbacks = [CSV_log, early_stop, best_model, MemoryCallback()], 
                epochs = int(parser.get('Training_Parameters', 'epochs')), 
                verbose = int(parser.get('Training_Parameters', 'verbose')),
                max_q_size = int(parser.get('Training_Parameters', 'max_queue_size'))
                )


#################### Saving and Calculation of Result for Test Dataset ######################

  print('\n Save the Model \n')
  model.save(os.path.join(\
  file_location,'train_hist/{}/{}/final_network.h5'.format(today,project_name)))  # save trained network

  print('\n Calculate Results... \n')
  res = []
  test_out = []

  for i in range(len(input_data)):
    print('Predict Values for File {}/{}'.format(i, len(input_data)))
    test  = input_data[i][test_inds[i][0]:test_inds[i][1]]
    test_out_chunk = np.log10(output_data[i][test_inds[i][0]:test_inds[i][1],0:1])
    res_chunk= model.predict(test, verbose=int(parser.get('Training_Parameters', 'verbose')))
    res.extend(list(res_chunk))
    test_out.extend(list(test_out_chunk))


  np.save(os.path.join(file_location,'train_hist/{}/{}/test_results.npy'.format(today, project_name)), 
    [res, np.squeeze(test_out)])

  print(' \n Finished .... ')
