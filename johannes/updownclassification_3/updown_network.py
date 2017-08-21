#!/usr/bin/env python
# coding: utf-8

# this script uses theos datasets. That is a try to reduce overfitting

import jkutils
from jkutils import zenith_to_binary, read_files
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32" 
os.environ["PATH"] += os.pathsep + '/usr/local/cuda/bin/'
os.environ['PYTHONUNBUFFERED'] = '1'
import sys
import inspect
import numpy as np
with jkutils.suppress_stdout_stderr(): #prevents printed info from theano
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
from sklearn.model_selection import train_test_split
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
import shutil

## constants ##
energy, azmiuth, zenith, muex = 0, 1, 2, 3

################# Function Definitions ####################################################################

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", help="The name for the Project", type=str ,default='updown_NN')
    parser.add_argument("--input", help="Name of the input files seperated by :", type=str ,default='all')
    parser.add_argument("--model", help="Name of the File containing the model", type=str, default='FCNN_v1.cfg')
    parser.add_argument("--using", help="charge or time", type=str, default='time')
    parser.add_argument("--virtual_len", help="Use an artifical array length (for debugging only!)", type=int , default=-1)
    parser.add_argument("--continue", help="Give a folder to continue the training of the network", type=str, default = 'None')
    parser.add_argument("--date", help="Give current date to identify safe folder", type=str, default = 'None')
    parser.add_argument("--version", action="version", version='%(prog)s - Version 3.0')
    parser.add_argument("--filesizes", help="Print the number of events in each file and don't do anything else.", nargs='?',
                       const=True, default=False)
    parser.add_argument("--testing", help="loads latest model and just does some testing", nargs='?', const=True, default=False)
    parser.add_argument("--crtfolders", help="creates the folderstructure so you can redirect nohup output to it. take care of day-change: at 7'o'clock in the morning (german summer time).", nargs='?', const=True, default=False)
      # Parse arguments
    args = parser.parse_args()
    return args

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
                add_layer(model, layer, args, kwargs)
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
        
        
def generator(batch_size, input_data, out_data, inds, inf_times_as = 1): 
    #even when using charge as input this could be left as it is (preprocess will not find any inf values)
    preprocess = jkutils.preprocess # this is needed, because python throws an error, if preprocess is used when using=time
                                    # (preprocess must then be in locals if referenced before.)
    if 'args' in globals():
        if args.using == 'charge':
            preprocess = jkutils.fake_preprocess
    batch_input = np.zeros((batch_size, 20,10,60,1))
    batch_out = np.zeros((batch_size,1))
    cur_file = 0
    cur_event_id = inds[cur_file][0]
    cur_len = 0
    up_to = inds[cur_file][1]
    print "here"
    print out_data[0][0]["zenith"]
    print "exiting"
    #sys.exit(0)
    while True:
        temp_in = []
        temp_out = []
        while cur_len < batch_size:
            fill_batch = batch_size-cur_len
            if fill_batch < (up_to-cur_event_id):
                temp_in.extend(input_data[cur_file][cur_event_id:cur_event_id+fill_batch])
                temp_out.extend(out_data[cur_file][cur_event_id:cur_event_id+fill_batch])
                cur_len += fill_batch
                cur_event_id += fill_batch
            else:
                temp_in.extend(input_data[cur_file][cur_event_id:up_to])
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
            batch_input[i] = preprocess(temp_in[i], replace_with = inf_times_as)
            batch_out[i] = zenith_to_binary(temp_out[i]["zenith"])
        cur_len = 0 
        """
        from scipy.stats import describe
        print describe(batch_input[0])
        print describe(np.array(batch_out))
        print batch_out[0:50]
        sys.exit()
        """
        #batch_input should be tuple: (samples, conv_dim1, conv_dim2, conv_dim3, channels)
        yield (batch_input, batch_out)
            

if __name__ == "__main__":

#################### Process Command Line Arguments ######################################

    currentfile = inspect.getfile(inspect.currentframe())
    config_path = os.path.join(os.path.dirname(os.path.abspath(currentfile)), 'config.cfg')
    parser = ConfigParser()
    parser.read(config_path)
    file_location = parser.get('Basics', 'thisfolder') # /data/user/jkager/NN_Reco/johannes/updownclassification_3/
    data_location = parser.get('Basics', 'data_enc_folder')
  
    args = parseArguments()
    print"\n ############################################"
    print("You are running the network script with arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))
    print"############################################\n "
  
    project_name = args.project
  
    input_files = jkutils.get_filenames(args.input)
  
    
#################### set today date and check for --filesizes #################################  
  
    if args.filesizes:
        read_files(input_files, data_location, printfilesizes=True)
        print 20*"-" + "\nOnly printed filesizes. Now exiting."
        sys.exit(1)
    
    if args.date != 'None':
        today = args.date
    else:
        today = datetime.date.today()
        

        
################### see if --continue is set and do accordingly ###############
    if not args.testing:
        if args.__dict__['continue'] != 'None':
            shelf = shelve.open(os.path.join(file_location,args.__dict__['continue'], 'run_info.shlf'))
            project_name = shelf['Project']
            input_files = shelf['Files'].split(':')
            if input_files[0] == 'all':
                input_files = os.listdir(os.path.join(data_location, 'training_data/'))
            train_inds = shelf['Train_Inds'] 
            valid_inds = shelf['Valid_Inds']
            test_inds = shelf['Test_Inds']
            try:
                args.using = shelf['using']
            except KeyError:
                pass
            model = load_model(os.path.join(file_location, args.__dict__['continue'], 'best_val_loss.npy'))
            today = args.__dict__['continue'].split('/')[1]
            print(today)
            shelf.close()
            
            config_path = os.path.join(file_location,args.__dict__['continue'], 'config.cfg')
            parser = ConfigParser()
            parser.read(config_path)
            
            input_data, out_data, file_len = read_files(input_files, data_location, using=args.using)
            
        else:
#################### Create Folder Strucutre #####################
            ## Create Folders
            folders=['train_hist/',
                     'train_hist/{}'.format(today),
                     'train_hist/{}/{}'.format(today, project_name)]
            for folder in folders:
                if not os.path.exists('{}'.format(os.path.join(file_location,folder))):
                    os.makedirs('{}'.format(os.path.join(file_location,folder)))
            
            if args.crtfolders:
                print "Created the folder structure for you:", folders[2]
                print "Now exiting."
                sys.exit(1)
                    
##################### Load and Split Dataset #########################################################
            input_data, out_data, file_len = read_files(input_files, data_location, using=args.using, virtual_len = args.virtual_len)

            tvt_ratio=[float(parser.get('Training_Parameters', 'training_fraction')),
                float(parser.get('Training_Parameters', 'validation_fraction')),
                float(parser.get('Training_Parameters', 'test_fraction'))] 
    
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
            shelf['Files'] = ':'.join(input_files)
            shelf['arguments'] = str(sys.argv)
            shelf['Train_Inds'] = train_inds
            shelf['Valid_Inds'] = valid_inds
            shelf['Test_Inds'] = test_inds
            shelf['using'] = args.using
            shelf['inf_times_as'] = float(parser.get('Training_Parameters','inf_times_as'))
            shelf.close()

            shutil.copy(args.model, os.path.join(file_location, 'train_hist/{}/{}/model.cfg'.format(today, project_name)))
            shutil.copy(config_path, os.path.join(file_location, 'train_hist/{}/{}/config.cfg'.format(today, project_name)))           
            
#################### Create Model #########################################################
            model = base_model(args.model)
            #from keras.utils import plot_model
            #plot_model(model, to_file='model.png')

#################### Train the Model ########################################################
        start = time.time()

        CSV_log = keras.callbacks.CSVLogger( \
          os.path.join(file_location,'train_hist/{}/{}/loss_logger.csv'.format(today, project_name)), 
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

        batch_size = int(parser.get('Training_Parameters', 'batch_size'))
        model.fit_generator(generator(batch_size, input_data, out_data, train_inds,
                                      float(parser.get('Training_Parameters','inf_times_as'))), 
                      steps_per_epoch = math.ceil(np.sum([k[1]-k[0] for k in train_inds])/batch_size),
                      validation_data = generator(batch_size, input_data, out_data, valid_inds, 
                                                  float(parser.get('Training_Parameters','inf_times_as'))),
                      validation_steps = math.ceil(np.sum([k[1]-k[0] for k in valid_inds])/batch_size),
                      callbacks = [CSV_log, early_stop, best_model, MemoryCallback()], 
                      epochs = int(parser.get('Training_Parameters', 'epochs')), 
                      verbose = int(parser.get('Training_Parameters', 'verbose')),
                      max_q_size=int(parser.get('Training_Parameters', 'max_queue_size'))
                      )

        print "time to fit: {:d}h {:d}min {:.2f}sec".format(*(lambda t: (int(t/3600),int(t/60)%60,t%60))(time.time()-start))
        
#################### Saving and Calculation of Result for Test Dataset ######################
  
    if args.testing:
        if not os.path.exists('{}'.format(os.path.join(file_location,'train_hist/'))):
            print "no training history found"
            sys.exit()
        hist_list = os.listdir(os.path.join(file_location,'train_hist/'))
        if len(hist_list) == 0:
            print "no training history found"
            sys.exit()
        for today in reversed(sorted(hist_list)):
            project_folder = 'train_hist/{}/{}'.format(today, project_name)
            print "looking for", project_folder
            if os.path.exists('{}'.format(os.path.join(file_location,project_folder))):
                print "project found!"
                break
        else:
            print "project not found. exiting..."
            sys.exit(-1)
        shelf = shelve.open(os.path.join(file_location, project_folder, 'run_info.shlf'))
        input_files = shelf['Files'].split(':')
        print "Input Files:", input_files
        if len(input_files) == 1: #this could be something like ['h01'] (inputformat)
            #try to decode fileinput format
            input_files = jkutils.get_filenames(input_files[0])
            print "decodes to: ", input_files
        for f in input_files:
            if not os.path.isfile(os.path.join(data_location, 'training_data/{}'.format(f))):
                print "file not found:", f
                print "exiting script."
                sys.exit(1)
        print "All files found. proceeding..."
        train_inds = shelf['Train_Inds'] 
        valid_inds = shelf['Valid_Inds']
        test_inds = shelf['Test_Inds']
        try:
            args.using = shelf['using']
        except KeyError:
            pass
        shelf.close()
        
        config_path = os.path.join(file_location, project_folder, 'config.cfg')
        parser = ConfigParser()
        parser.read(config_path)
            
        input_data, out_data, file_len = read_files(input_files, data_location, using=args.using)
        print('\n Load the Model (final_network.h5) \n')
        model = load_model(os.path.join(\
        file_location,'train_hist/{}/{}/final_network.h5'.format(today, project_name)))
    else:
        print('\n Save the Model \n')
        model.save(os.path.join(\
        file_location,'train_hist/{}/{}/final_network.h5'.format(today, project_name)))  # save trained network
  
    print('\n Calculate Results... \n')
    res = []
    test_out = []
  
    inf_times_as=float(parser.get('Training_Parameters','inf_times_as'))
    if args.using == 'charge':
        preprocess = jkutils.fake_preprocess
    else:
        preprocess = jkutils.preprocess
    for i in range(len(input_data)):
        print('Predict Values for {}'.format(input_files[i]))
        test_in_chunk  = preprocess(input_data[i][test_inds[i][0]:test_inds[i][1]], 
                                    replace_with = inf_times_as)
        test_out_chunk = zenith_to_binary(out_data[i][test_inds[i][0]:test_inds[i][1],"zenith"])
        res_chunk = model.predict(test_in_chunk, verbose=int(parser.get('Training_Parameters', 'verbose')))
        res.extend(list(res_chunk))
        test_out.extend(list(test_out_chunk))
  
    res = np.squeeze(res)
    test_out = np.squeeze(test_out)
    np.save(os.path.join(file_location,'train_hist/{}/{}/test_results.npy'.format(today, project_name)), 
      [res, test_out])
    correct = np.sum(np.round(res) == test_out)
    total = len(res)
    print "{} / {} = {:6.2f}%".format(correct, total, float(correct)/total*100)
  
    print(' \n Finished .... ')
