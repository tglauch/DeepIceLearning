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
import h5py
import resource

def read_files(file_location, input_files, virtual_len=-1):

  """Create an Array of Input and Output HDF5 Monte Carlo data from a given list of files(e.g file1:file2 ...) 

  Arguments:
  file_location: location of this file, as saved in config.cfg
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
    hfile = h5py.File(data_file, 'r')
    if virtual_len == -1:
      ## Bad style of doing this.
      ## For further version better save the shape directly in the File
      ### Which also solves the problem of checking for matching input size of the files
      data_len = len(hfile[hfile.keys()[0]])  
    else:
      data_len = virtual_len
      print('Only use the first {} Monte Carlo Events'.format(data_len))

    input_data.append(hfile)
    output_data.append(h5py.File(data_file, 'r'))
    file_len.append(data_len)

  return input_data, output_data, file_len

def prepare_input(one_input_array, model_settings):

  """given the transformations of the 'raw' input data (e.g. the charges per gridelement) the
  the input shape for the network is calculated. example: transformation is np.sum(x) -->
  input shape = (1,)


  Arguments:
  one_input_array : one exemplary input data array on which the transformation is applied
  model_settings : the different settings for the model. has to include a section [Inputs] 
                   otherwise assume no transformation at all 

  Returns: 
  shapes : the shapes after transformation (ordered)
  shape_names : the names of the correspondincxg model branches (ordered)

  """

  print(model_settings)
  inp_variables = []
  transformations = []
  shapes = []
  shape_names = []
  mode = ''
  if len(model_settings) == 0:
      model_settings = [['{Inputs}'], ['[model]', 'variables = [charge]', 'transformations = [x]']]
  for block in model_settings:
      if block[0][0]=='{' and block[0][-1]=='}':
          mode=block[0][1:-1]
          continue
      if mode == 'Inputs':
          for element in block:
              if element[0]=='[' and element[-1]==']':
                  shape_names.append(element[1:-1])
              else:
                  split_row = element.split('=')
                  if split_row[0].strip() == 'variables':
                      inp_variables.append(split_row[1].strip().split(','))
                  elif split_row[0].strip() == 'transformations':
                      transformations.append(split_row[1].strip().split(','))
      else:
        ### Todo: Once the output shape gets more flexible, change this here. function is then called
        ## prepare_inp_out_shapes
          continue

  for i in range(len(shape_names)):
      temp_shape_arr = []
      for j in range(len(inp_variables[i])):
          temp_shape_arr.append(np.shape(eval(transformations[i][j].replace('x', 'one_input_array[\'{}\'][0]'.format(inp_variables[i][j])))))
      if all(x==temp_shape_arr[0] for x in temp_shape_arr):
        shapes.append(temp_shape_arr[0][0:-1]+(len(temp_shape_arr),))
      else:
        raise Exception('The transformations that you have applied do not results in the same shape!!!!')
      print shapes
  return shapes, shape_names, inp_variables, transformations

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

def generator(batch_size, input_data, output_data, inds,
  inp_shape, inp_variables, inp_transformations,
  out_shape=[(1,)], out_variables = [0], out_transformations= ['np.log10(x)']):

  """ This function is a real braintwister and presumably really bad implemented.
  It produces all input and output data and applies the transformations
  as defined in the network definition file.

  Arguments:
  batch size : the batch size per gpu
  input_data: a list containing the input data
  output_data: a list containing the output data
  inds: the index range used for the training set
  inp_shape: the shapes of the input neurons
  inp_variables: list of variables used for the input
  inp_transformations: list of transformations for the input data
  out_shape: shape of the output data
  out_variables: variables for the output data
  out_transformations: transformations applied to the output data


  Returns: 
  batch_input : a batch of input data
  batch_out: a batch of output data

  """

  batch_input = [ np.zeros((batch_size,)+i) for i in inp_shape ]
  ###Todo: Read correct output varibles from the model configuration
  ### for this the dtype in the Create Dataset file has to be changed 
  batch_out = np.zeros((batch_size,1))
  cur_file = 0
  cur_event_id = inds[cur_file][0]
  cur_len = 0
  up_to = inds[cur_file][1]
  loop_counter = 0 
  temp_out = []
  temp_in = []
  while True:
    loop_counter+=1
    if (loop_counter%500)==1:
      print(' \n RAM Usage {:.2f} GB \n \n'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))
    for j, var_array in enumerate(inp_variables):
      for k, var in enumerate(var_array):
        temp_cur_file = cur_file
        temp_cur_event_id = cur_event_id
        temp_up_to = up_to
        cur_len = 0
        while cur_len<batch_size:
          fill_batch = batch_size-cur_len
          if fill_batch < (temp_up_to-cur_event_id):
            temp_in.extend(input_data[temp_cur_file][var][temp_cur_event_id:temp_cur_event_id+fill_batch])
            if j==0 and k==0:
              temp_out.extend(output_data[temp_cur_file]['reco_vals'][temp_cur_event_id:temp_cur_event_id+fill_batch])
            cur_len += fill_batch
            temp_cur_event_id += fill_batch
          else:
            temp_in.extend(input_data[temp_cur_file][var][temp_cur_event_id:temp_up_to])
            if j==0 and k==0:
              temp_out.extend(output_data[temp_cur_file]['reco_vals'][temp_cur_event_id:temp_up_to])
            cur_len += temp_up_to-temp_cur_event_id
            temp_cur_file+=1
            print 'Read File Number {}'.format(temp_cur_file) 
            if temp_cur_file == len(inds):
              break
            else:
              temp_cur_event_id = inds[temp_cur_file][0]
              temp_up_to = inds[temp_cur_file][1]

        for i in range(len(temp_in)):
          slice_ind = [slice(None)]*batch_input[j][i].ndim
          slice_ind[-1] = slice(k,k+1,1)
          pre_append = eval(inp_transformations[j][k].replace('x', 'temp_in[i]'))
          if var == 'time':
            pre_append[pre_append==np.inf]=-1
          batch_input[j][i][slice_ind] = pre_append 
        temp_in = []

    for j, var in enumerate(out_variables):
      for i in range(len(temp_out)):
        batch_out[i][j] =  eval(out_transformations[j].replace('x', 'temp_out[i][var]')) 
    temp_out=[]

    if temp_cur_file == len(inds):
      cur_file = 0
      cur_event_id = inds[0][0]
      up_to = inds[0][1] 
    else:
      cur_file = temp_cur_file
      cur_event_id = temp_cur_event_id
      up_to = temp_up_to    
    if (loop_counter%500)==1:
      print(' \n RAM Usage {:.2f} GB \n \n'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))
    yield (batch_input, batch_out)
