#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse
from six.moves import configparser
#changed this because ConfigParser was not available on the RZ in Aachen
#from configparser import ConfigParser
import datetime
from workload_managers import *
import shelve

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main_config",
        help="absolute path to main config file",
        type=str)
    parser.add_argument(
        "--folder",
        type=str, required=True,
        help="folder where all the config are saved")
    parser.add_argument(
        "--batch_size",
        type=int, default='64',
        help="the batch size")
    parser.add_argument(
        "--model", type=str,
        help="absolute path to the model file")
    parser.add_argument(
        "--weights", type=str,
        help="absolute path to the weights file")
    parser.add_argument(
        "--gpus", type=int,
        help="number of gpus to be used")
    parser.add_argument(
        "--memory",
        help="specify the RAM requirements",
        type=int, default=-1)
    parser.add_argument(
        "--test_data",
        help="test data",
        type=str)
    parser.add_argument(
        "--exp_data",
        help="path to experimental data",
        type=str)    
    parser.add_argument(
        "--outfile", type=str)
    args = parser.parse_args()
    return args



args = parseArguments().__dict__
parser = configparser.ConfigParser()
if args['main_config'] == None:
    args['main_config'] = os.path.join(args['folder'], 'config.cfg')
parser.read(args["main_config"])
parser_dict = {s:dict(parser.items(s)) for s in parser.sections()}

train_location = parser.get('Basics', 'train_folder')
workload_manager = parser.get('Basics', 'workload_manager')
if args['gpus'] == None:
    request_gpus = parser.get('GPU', 'request_gpus')
else:
    request_gpus = args['gpus']

if args['weights'] == None:
    args['weights'] = os.path.join(args['folder'], 'best_val_loss.npy')
if args['model'] == None:
    args['model'] = os.path.join(args['folder'], 'model.py')
if args['memory'] == -1:
    request_memory = parser.get('GPU', 'request_memory')
else:
    request_memory = args['memory']
requirements = parser.get('GPU', 'requirements')
thisfolder = parser.get("Basics", "thisfolder")
if 'exclude_node' in parser_dict['GPU'].keys():
    exclude = parser.get('GPU', 'exclude_apply')
else:
    exclude = ' '

if workload_manager not in ['slurm','condor','bsub']:
    raise NameError(
        'Workload manager not defined!')


arguments = ' --main_config {}  --folder {} --batch_size {} --model {} --weights {}'.format(\
                        args['main_config'], args['folder'], args['batch_size'], args['model'],args['weights'] )

if args['exp_data']:
    arguments += ' --exp_data {}'.format(args['exp_data'])
elif args['test_data'] is not None:
    arguments += ' --test_data {}'.format(args['test_data'])
else:
    raise ValueError('No data provided')
if args['outfile'] is not None:
    arguments += ' --outfile {}'.format(args['outfile'])
save_path = args['folder']
condor_out_folder = os.path.join(save_path, 'condor')
if workload_manager == 'slurm':
    submit_info = make_slurm("apply_env.sh",\
                             request_gpus,\
                             float(request_memory) * 1e3,\
                             condor_out_folder,\
                             train_location,\
                             arguments,\
                             thisfolder,\
                             exclude)
elif workload_manager == 'condor':
    submit_info = make_condor("apply_env.sh",\
                              request_gpus,\
                              request_memory,\
                              requirements,\
                              condor_out_folder,\
                              arguments,\
                              thisfolder)
elif workload_manager == 'bsub':
    submit_info = make_bsub("apply_env.sh",\
                            request_memory,\
                            condor_out_folder,\
                            thisfolder,\
                            arguments,\
                            request_cpus=1)


print(submit_info)
submitfile_full = os.path.join(condor_out_folder, 'submit.sub')
with open(submitfile_full, "wc") as file:
    file.write(submit_info)

if workload_manager == 'slurm':
    os.system("sbatch {}".format(submitfile_full))
elif workload_manager =="condor":
    os.system("condor_submit {}".format(submitfile_full))
elif workload_manager =="bsub":
    os.system("bsub<{}".format(submitfile_full))
time.sleep(3)
