#!/usr/bin/env python
# coding: utf-8

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32" 
os.environ["PATH"] += os.pathsep + '/usr/local/cuda/bin/'
import sys
import numpy as np
import theano
# theano.config.device = 'gpu'
# theano.config.floatX = 'float32'
import keras
from keras.models import Sequential, load_model
import argparse
from configparser import ConfigParser
import h5py
import shelve



def parseArguments():

  parser = argparse.ArgumentParser()
  parser.add_argument("--folder", help="The path to the project file", type=str)
  parser.add_argument("--final", dest='final', action='store_true')
  parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')
    # Parse arguments
  args = parser.parse_args()
  return args

if __name__ == "__main__":

#################### Process Command Line Arguments ######################################

  parser = ConfigParser()
  parser.read('config.cfg')
  file_location = parser.get('Basics', 'thisfolder')

  args = parseArguments()
  print"\n ############################################"
  print("You are running the script with arguments: ")
  for a in args.__dict__:
      print(str(a) + ": " + str(args.__dict__[a]))
  print"############################################\n "
  
#################### Load and Split the Datasets ######################################  

  DATA_DIR = os.path.join(file_location, args.__dict__['folder'])

  shelf = shelve.open(os.path.join(DATA_DIR, 'run_info.shlf'))

  if shelf['Files']=='all':
    input_files = os.listdir(os.path.join(file_location, 'training_data/'))
  else:
  	input_files = (shelf['Files']).split(':')
  input_data = []
  out_data = []

  for run, input_file in enumerate(input_files):
    data_file = os.path.join(file_location, 'training_data/{}'.format(input_file))
    input_data.append(h5py.File(data_file, 'r')['charge'])
    out_data.append(h5py.File(data_file, 'r')['reco_vals'])

  if args.__dict__['final']:
  	model = load_model(os.path.join(DATA_DIR,'final_network.h5'))
  else:
  	model = load_model(os.path.join(DATA_DIR,'best_val_loss.npy'))  	
  res = []
  test_out = []

  test_inds =   shelf['Test_Inds']

  for i in range(len(input_data)):
	print('Predict Values for {}'.format(input_file))
	down = test_inds[i][0]
	up = test_inds[i][1]
	split_list = range(down,up,2000)
	split_list.append(up)	
	for j in range(len(split_list)-1):
		test  = input_data[i][split_list[j]:split_list[j+1]]
		test_out_chunk = np.log10(out_data[i][split_list[j]:split_list[j+1],0:1])
		res_chunk= model.predict(test, verbose=1)
		res.extend(list(res_chunk))
		test_out.extend(list(test_out_chunk))


  np.save(os.path.join(DATA_DIR, 'test_res.npy'), [res, np.squeeze(test_out)])