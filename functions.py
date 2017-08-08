#!/usr/bin/env python
# coding: utf-8


import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D,\
 BatchNormalization, MaxPooling2D,Convolution3D,MaxPooling3D, Merge
from keras import regularizers
import os
import numpy as np
import h5py

def read_files(file_location, input_files, virtual_len=-1):

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

  """given the transformations of the 'raw' input data (i.e the charges per gridelement) the
  the input shape for the network is calculated. example: transformation is np.sum(x) -->
  input shape = (1,)


  Arguments:
  one_input_array : one exemplary input data array on which the transformation is applied
  model_settings : the different settings for the model. has to include a section [Inputs] 
                   otherwise assume no transformation at all 

  Returns: 
  shapes : the shapes after transformation (ordered)
  shape_names : the names of the corresponding model branches (ordered)

  """
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
  """Function that parses the config file and returns the settings and architecture of the model

  Arguments:
  conf_file_path : name of the config file in the folder ./Networks

  Returns: 
  settings : settings for the network. e.g. the input shapes
  model : the 'architecture' of the model. basically contains a list of layers with their args and kwargs
  """

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
  model_def : list containing blocks (list) of model-branches and layer definitions (compare the network config files)
  shape, shape_names: input shapes and names as constructed in prepare_input_shapes(one_input_array, model_settings)

  Returns: 
  model : the (non-compiled) model object

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
