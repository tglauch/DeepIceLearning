#!/usr/bin/env python
# coding: utf-8

import os
import sys
#from configparser import ConfigParser
from six.moves import configparser
import socket
import argparse
import lib.model_parse as mp
import cPickle as pickle
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.join(os.path.abspath(".."),'lib'))
import transformations

def parseArguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        help="The absolute path to the project file", type=str)
    parser.add_argument(
        "--test_data",
        help="Folder with test data files", type=str)
    parser.add_argument(
        "--exp_data", type=str,  nargs="+")
    parser.add_argument(
        "--outfile", type=str,
        required=False)
    parser.add_argument(
        "--main_config", type=str,
        help="Config file")
    parser.add_argument(
        "--batch_size", dest='batch_size',
        type=int, default=50)
    parser.add_argument(
        "--ngpus",
        help="number of GPUs", default=1)
    parser.add_argument(
        "--model",
        help="name of the model to apply")
    parser.add_argument(
        "--weights",
        help="weights of the model to apply")
    args = parser.parse_args().__dict__
    return args


args = parseArguments()
parser = configparser.ConfigParser()
DATA_DIR = args['folder']

if args['model'] == None:
        args['model'] = os.path.join(DATA_DIR, 'model.py')

if args['main_config'] == None:
    args['main_config'] = os.path.join(DATA_DIR, 'config.cfg')

if args["weights"] == None:
    args["weights"] = os.path.join(DATA_DIR, "best_val_loss.npy")
args["load_weights"] = args["weights"]

print args['main_config']
try:
    parser.read(args['main_config'])
except Exception:
    raise Exception('Config File is missing!!!!')

backend = parser.get('Basics', 'keras_backend')
os.environ["KERAS_BACKEND"] = backend
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

print os.environ['LD_LIBRARY_PATH'].split(os.pathsep)
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if backend == 'tensorflow':
    print('Run with backend Tensorflow')
    import tensorflow as tf
else:
    raise NameError('Backend {} currently not supported'.format(backend))

import numpy as np
import keras
from keras.models import Sequential, load_model
import h5py
from lib.functions import generator_v2
import math
import numpy.lib.recfunctions as rfn
from lib.functions import read_NN_weights, read_input_len_shapes

if __name__ == "__main__":

    # Process Command Line Arguments ######################################

    mc_location = parser.get("Basics", "mc_path")

    print"\n ############################################"
    print("You are running the script with arguments: ")
    for a in args:
        print(str(a) + ": " + str(args[a]))
    print"############################################\n "

    

    print('Make prediction for model in {}'.format(DATA_DIR))
    print DATA_DIR

    # with numpy dict
    run_info = np.load(os.path.join(DATA_DIR, 'run_info.npy'))[()]
    if args['exp_data'] is None:
        if args['test_data'] is not None:
            input_files= [os.path.join(args['test_data'], i) for i in os.listdir(args['test_data']) if i[-3:]=='.h5']
            file_len = read_input_len_shapes('', input_files)
            test_inds = [(0, tot_len)for tot_len in file_len]
        else:
            test_inds = run_info["Test_Inds"]
            if run_info['Files'] == ['all']:
                input_files = os.listdir(mc_location)
            elif isinstance(run_info["Files"], list):
                input_files = run_info['Files']
            else:
                input_files = run_info['Files'].split(':')
    else:
        input_files = np.concatenate([[os.path.join(i, j) for j in os.listdir(i)]
                                       if os.path.isdir(i) else [i] for i in args['exp_data']])
    conf_model_file = args['model']
    print('Input Files:')
    print(input_files)
    base_model = mp.parse_functional_model(conf_model_file,
            os.path.join(mc_location, input_files[0]), only_model=True)
    inp_shapes = run_info['inp_shapes']
    out_shapes = run_info['out_shapes']
    inp_trans = run_info['inp_trans']
    out_trans = run_info['out_trans']
    base_model = base_model.model(inp_shapes, out_shapes)
    ngpus = args['ngpus']

    print'Use {} GPUS'.format(ngpus)
    if ngpus > 1:
        model_serial = read_NN_weights(args, base_model)
        model = multi_gpu_model(model_serial, gpus=ngpus)
    else:
        model = read_NN_weights(args, base_model)
        model_serial = model
 
    os.system("nvidia-smi")

    # Saving the Final Model and Calculation/Saving of Result for Test Dataset ####
    if args['exp_data'] is None:
        use_data = False
        file_handlers = [os.path.join(mc_location, file_name)
                         for file_name in input_files]
        num_events = np.sum([k[1] - k[0] for k in test_inds])
        print('Apply the NN to {} events'.format(num_events))
    else:
        use_data = True
        file_handlers = input_files
        file_len = read_input_len_shapes('', input_files)
        print file_len
        test_inds = [(0, tot_len) for tot_len in file_len]

    steps_per_epoch = int(np.sum([math.ceil((1.*(k[1]-k[0])/args['batch_size']))
                                   for k in test_inds]))

    if steps_per_epoch == 0:
        print "steps per epoch is 0, therefore manually set to 1"
        steps_per_epoch = 1
    prediction = model.predict_generator(
        generator_v2(args['batch_size'],
                    file_handlers, test_inds,
                    inp_shapes, inp_trans,
                    out_shapes, out_trans,
                    use_data=use_data,
                    equal_len=False),
        steps=steps_per_epoch,
        verbose=1,
        max_queue_size=5,
        use_multiprocessing=False)
    mc_truth = [[] for br in out_shapes.keys()
                for var in out_shapes[br].keys()
                if var != 'general']
    reco_vals = None
    for i, file_handler_p in enumerate(file_handlers):
        down = test_inds[i][0]
        up = test_inds[i][1]
        file_handler = h5py.File(file_handler_p, 'r')
        temp_truth = file_handler['reco_vals'][down:up]
        if args['exp_data'] is None:
            for j, var in enumerate([var for br in out_shapes.keys()
                                    for var in out_shapes[br].keys()
                                    if var != 'general']):
                mc_truth[j].extend(temp_truth[var])
        if reco_vals == None:
            reco_vals = temp_truth
        else:
            reco_vals = np.concatenate([reco_vals, temp_truth])

    if args['exp_data'] is None:
        dtype = np.dtype([(var + '_truth', np.float64)
                          for br in out_shapes.keys()
                          for var in out_shapes[br].keys()
                          if var != 'general'])
        mc_truth = np.array(zip(*np.array(mc_truth)), dtype=dtype)

    save_name = args["weights"][:-4] + "_pred.pickle"
    print save_name
    if args['outfile'] is None:
        o_file = os.path.join(DATA_DIR, save_name)
    else:
        o_file = args['outfile']
    if args['exp_data'] is None:
        pickle.dump({"mc_truth": mc_truth,
                     "prediction": prediction,
                     "reco_vals": reco_vals},
                     open(o_file, "wc"))
    else:
        print('The prediction is {}'.format(prediction))
        pickle.dump({"prediction": prediction,
                     "reco_vals": reco_vals},
                     open(o_file, "wc"))
    print(' \n Finished .... Exiting.....')
    exit(0)
