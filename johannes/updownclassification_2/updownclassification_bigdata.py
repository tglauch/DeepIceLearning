#!/usr/bin/env python
# coding: utf-8

# bigdata == suggests that this script uses theos datasets. That is a try to reduce overfitting

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
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D,\
 BatchNormalization, MaxPooling2D,Convolution3D,MaxPooling3D
from keras.optimizers import SGD, Adagrad
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils import np_utils
from keras.utils.io_utils import HDF5Matrix
from keras.constraints import maxnorm
from keras import regularizers
from sklearn.cross_validation import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import h5py
import datetime
import gc
from ConfigParser import ConfigParser
import argparse
import tables
import math
import time
import resource
import shelve
import itertools

## constants ##
energy, azmiuth, zenith, muex = 0, 1, 2, 3

################# Function Definitions ####################################################################

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", help="The name for the Project", type=str ,default='updown_NN')
    parser.add_argument("--input", help="Name of the input files seperated by :", type=str ,default='all')
    parser.add_argument("--model", help="Name of the File containing the model", type=str, default='simple_FCNN.cfg')
    parser.add_argument("--virtual_len", help="Use an artifical array length (for debugging only!)", type=int , default=-1)
    parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')
    parser.add_argument("--filesizes", help="Print the number of events in each file and don't do anything else.", nargs='?',
                       const=True, default=False)
      # Parse arguments
    args = parser.parse_args()
    return args

def zenith_to_binary(zenith):
    return 1 if zenith > 1.5707963268 else 0

def add_layer(model, layer, args, kwargs):
    eval('model.add({}(*args,**kwargs))'.format(layer))
    return

def base_model(model_def):
    model = Sequential()
    with open(os.path.join(file_location,model_def)) as f:
        args = []
        kwargs = dict()
        layer = ''
        mode = 'args'
        for line in f:
            cur_line = line.strip()
            if cur_line == '' and layer != '':
                add_layer(model, layer, args,kwargs)
                args = []
                kwargs = dict()
                layer = ''
            elif cur_line[0] == '#':
                continue
            elif cur_line == '[kwargs]':
                mode = 'kwargs'
            elif layer == '':
                layer = cur_line[1:-1]
            elif mode == 'args':
                try:
                    args.append(eval(cur_line.split('=')[1]))
                except:
                    args.append(cur_line.split('=')[1])
            elif mode == 'kwargs':
                split_line = cur_line.split('=')
                try:
                    kwargs[split_line[0].strip()] = eval(split_line[1].strip())
                except:
                    kwargs[split_line[0].strip()] = split_line[1].strip()
        if layer != '':
            add_layer(model, layer, args,kwargs)
    
    print(model.summary())
    #adam = keras.optimizers.Adam(lr=float(parser.get('Training_Parameters', 'learning_rate')))
    adagrad = Adagrad()
    model.compile(loss='binary_crossentropy', optimizer=adagrad, metrics=['accuracy'])
    return model

class MemoryCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, log={}):
        print('RAM Usage {:.2f} GB'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))

def generator(batch_size, input_data, out_data, inds):
    batch_input = np.zeros((batch_size, 1, 21, 21, 51))
    batch_out = np.zeros((batch_size,1))
    cur_file = 0
    cur_event_id = inds[cur_file][0]
    cur_len = 0
    up_to = inds[cur_file][1]
    while True:
        temp_in = []
        temp_out = []
        while cur_len<batch_size:
            fill_batch = batch_size-cur_len
            if fill_batch < (up_to-cur_event_id):
                temp_in.extend(map(np.ndarray.flatten, input_data[cur_file][cur_event_id:cur_event_id+fill_batch]))
                temp_out.extend(out_data[cur_file][cur_event_id:cur_event_id+fill_batch])
                cur_len += fill_batch
                cur_event_id += fill_batch
            else:
                temp_in.extend(map(np.ndarray.flatten, input_data[cur_file][cur_event_id:up_to])
                temp_out.extend(out_data[cur_file][cur_event_id:up_to])
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
            batch_input[i] = temp_in[i]
            batch_out[i] = zenith_to_binary(temp_out[i][zenith])
        cur_len = 0 
        yield (batch_input, batch_out)


if __name__ == "__main__":

#################### Process Command Line Arguments ######################################

    parser = ConfigParser()
    parser.read('config.cfg')
    file_location = parser.get('Basics', 'thisfolder')
    data_location = parser.get('Basics', 'data_enc_folder')
  
    args = parseArguments()
    print"\n ############################################"
    print("You are running the script with arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))
    print"############################################\n "
  
    project_name = args.project
  
    if args.input =='all':
        input_files = os.listdir(os.path.join(data_location, 'training_data/'))
    else:
        input_files = (args.input).split(':')
  
    
#################### Load and Split the Datasets ######################################  
  
    tvt_ratio=[float(parser.get('Training_Parameters', 'training_fraction')),
    float(parser.get('Training_Parameters', 'validation_fraction')),
    float(parser.get('Training_Parameters', 'test_fraction'))] 
  
    ## Create Folders
    today = datetime.date.today()
    folders=['train_hist/',
     'train_hist/{}'.format(today),
     'train_hist/{}/{}'.format(today, project_name)]
    for folder in folders:
        if not os.path.exists('{}'.format(os.path.join(file_location,folder))):
            os.makedirs('{}'.format(os.path.join(file_location,folder)))
  
    input_data = []
    out_data = []
    file_len = []
  
    for run, input_file in enumerate(input_files):
        data_file = os.path.join(data_location, 'training_data/{}'.format(input_file))
  
        if args.__dict__['virtual_len'] == -1:
            data_len = len(h5py.File(data_file)['time'])
        else:
            data_len = args.virtual_len
            print('Only use the first {} Monte Carlo Events'.format(data_len))
        if args.filesizes:
            print "{:10d}   {}".format(data_len, input_file)
        else:
            input_data.append(h5py.File(data_file, 'r')['time'])
            out_data.append(h5py.File(data_file, 'r')['reco_vals'])
            file_len.append(data_len)
            # print input_data[-1].shape = (970452, 1, 21, 21, 51)
    if args.filesizes:
        print 20*"-" + "\nOnly printed filesizes. Now exiting."
        sys.exit()
    train_frac  = float(tvt_ratio[0])/np.sum(tvt_ratio)
    valid_frac = float(tvt_ratio[1])/np.sum(tvt_ratio)
    train_inds = [(0, int(tot_len*train_frac)) for tot_len in file_len] 
    valid_inds = [(int(tot_len*train_frac), int(tot_len*(train_frac+valid_frac))) for tot_len in file_len] 
    test_inds = [(int(tot_len*(train_frac+valid_frac)), tot_len-1) for tot_len in file_len] 
  
    print(train_inds)
    print(valid_inds)
    print(test_inds)
  
#################### Save Run-Information #################################################
  
    shelf = shelve.open(os.path.join(file_location,'./train_hist/{}/{}/run_info.shlf'.format(today, project_name)))
    shelf['Project'] = project_name
    shelf['Files'] = args.input
    shelf['Train_Inds'] = train_inds
    shelf['Valid_Inds'] = valid_inds
    shelf['Test_Inds'] = test_inds
    shelf.close()
  
#################### Train the Model #########################################################
  
    CSV_log = keras.callbacks.CSVLogger( \
      os.path.join(file_location,'./train_hist/{}/{}/loss_logger.csv'.format(today, project_name)), 
      append=True)
  
    """ see: https://keras.io/callbacks/#earlystopping """
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
  
    model = base_model(args.model)
    batch_size = int(parser.get('Training_Parameters', 'batch_size'))
    model.fit_generator(generator(batch_size, input_data, out_data, train_inds), 
                  steps_per_epoch = math.ceil(np.sum([k[1]-k[0] for k in train_inds])/batch_size),
                  validation_data = generator(batch_size, input_data, out_data, valid_inds),
                  validation_steps = math.ceil(np.sum([k[1]-k[0] for k in valid_inds])/batch_size),
                  callbacks = [CSV_log, early_stop, best_model, MemoryCallback()], 
                  epochs = int(parser.get('Training_Parameters', 'epochs')), 
                  verbose = int(parser.get('Training_Parameters', 'verbose')),
                  max_q_size=int(parser.get('Training_Parameters', 'max_queue_size'))
                  )
  
  
#################### Saving and Calculation of Result for Test Dataset ######################
  
    print('\n Save the Model \n')
    model.save(os.path.join(\
    file_location,'train_hist/{}/{}/final_network.h5'.format(today, project_name)))  # save trained network
  
    print('\n Calculate Results... \n')
    res = []
    test_out = []
  
    for i in range(len(input_data)):
        print('Predict Values for {}'.format(input_files[i]))
        test  = input_data[i][test_inds[i][0]:test_inds[i][1]]
        test_out_chunk = zenith_to_binary(out_data[i][test_inds[i][0]:test_inds[i][1],zenith:zenith+1])
        res_chunk = model.predict(test, verbose=int(parser.get('Training_Parameters', 'verbose')))
        res.extend(list(res_chunk))
        test_out.extend(list(test_out_chunk))
  
    res = np.array(res)
    test_out = np.squeeze(test_out)
    
    np.save(os.path.join(file_location,'train_hist/{}/{}/test_results.npy'.format(today, project_name)), 
      [res, test_out])
    
    correct = np.sum(res == test_out)
    total = len(res)
    print "{} / {} = {:6.2f}".format(correct, total, float(correct)/total*100)
  
    print(' \n Finished .... ')
