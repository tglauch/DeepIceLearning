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

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main_config",
        help="main config file, user-specific",
        type=str, default='default.cfg')
    parser.add_argument(
        "--project",
        help="The name for the Project",
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
        help="Provide a folder to continue the training of the network",
        type=str, default='None')
    parser.add_argument(
        "--load_weights",
        help="Provide a path to pre-trained model weights",
        type=str, default='None')
    parser.add_argument(
        "--version", action="version",
        version='%(prog)s - Version 1.0')
    parser.add_argument(
        "--apply_test",
        action="store_true",\
        help="Apply test after training? ")
    args = parser.parse_args()
    return args



args = parseArguments().__dict__
parser = configparser.ConfigParser()
if args['continue'] != 'None':
    parser.read(os.path.join(args["continue"], 'config.cfg'))
    args["main_config"] = os.path.join(args["continue"], 'config.cfg')
else:
    parser.read(args["main_config"])
parser_dict = {s:dict(parser.items(s)) for s in parser.sections()}

train_location = parser.get('Basics', 'train_folder')
workload_manager = parser.get('Basics', 'workload_manager')
request_gpus = parser.get('GPU', 'request_gpus')
request_memory = parser.get('GPU', 'request_memory')
requirements = parser.get('GPU', 'requirements')
project_name = args['project']
thisfolder = parser.get("Basics", "thisfolder")
model_name = parser.get("Basics", "model_name")
if 'exclude_node' in parser_dict['GPU'].keys():
    exclude = parser.get('GPU', 'exclude_node')
else:
    exclude = 'nothing'

if workload_manager not in ['slurm','condor','bsub']:
    raise NameError(
        'Workload manager not defined. Should either be condor or slurm!')

files = args['input']

arguments = ''
for a in args:
    if a == 'input':
        arguments += ' --input {} '.format(files)
    elif a =="apply_test":
        if args[a]:
            pass
    else:
        arguments += ' --{} {} '.format(a, args[a])

if args['continue'] != 'None':
    #arguments += '--continue {}'.format(args['continue'])
    save_path = args['continue']
    condor_out_folder = os.path.join(args['continue'], 'condor')
    #args["model"] = os.path.join(args['continue'], 'model.py')
    args["model"] = os.path.join(args['continue'], model_name)
else:
    today = str(datetime.datetime.now()).\
        replace(" ", "-").split(".")[0].replace(":", "-")
    if 'save_path' in parser_dict['Basics'].keys():
	save_path = parser.get('Basics', 'save_path')
    else:
        save_path = os.path.join(
            train_location, '{}/{}'.format(project_name, today))
    condor_out_folder = os.path.join(save_path, 'condor')

    if not os.path.exists(condor_out_folder):
        print('Create Condor-Out Folder: \n {}'.format(condor_out_folder))
        os.makedirs(condor_out_folder)
    arguments += ' --save_folder {} '.format(save_path)

arguments += ' --ngpus {} '.format(request_gpus)
if workload_manager == 'slurm':
    if args['continue'] != 'None':
                submit_info = make_slurm("NN_continue.sh",\
                             request_gpus,\
                             float(request_memory) * 1e3,\
                             condor_out_folder,\
                             train_location,\
                             arguments,\
                             thisfolder,\
                             exclude)
    else:
        submit_info = make_slurm("NN.sh",\
                             request_gpus,\
                             float(request_memory) * 1e3,\
                             condor_out_folder,\
                             train_location,\
                             arguments,\
                             thisfolder,\
                             exclude)
elif workload_manager == 'condor':
    submit_info = make_condor("Neural_Network.py",\
                              request_gpus,\
                              request_memory,\
                              requirements,\
                              condor_out_folder,\
                              arguments,\
                              thisfolder)
elif workload_manager == 'bsub':
    submit_info = make_bsub("Neural_Network.py",\
                            request_memory,\
                            condor_out_folder,\
                            thisfolder,\
                            arguments,\
                            apply_test = args["apply_test"],\
                            request_cpus=22,
                            cfg_file = args["main_config"],
                            save_path = save_path)


print(submit_info)
submitfile_full = os.path.join(condor_out_folder, 'submit.sub')
with open(submitfile_full, "wc") as file:
    file.write(submit_info)

if not os.path.exists(os.path.join(save_path, 'config.cfg')):
    os.system("cp {} {} ".format(
        args["main_config"], os.path.join(save_path, 'config.cfg')))
if not os.path.exists(os.path.join(save_path, model_name)):
#if not os.path.exists(os.path.join(save_path, 'model.py')):
    os.system("cp {} {} ".format(
        #args["model"], os.path.join(save_path, 'model.py')))
        args["model"], os.path.join(save_path, model_name)))

if workload_manager == 'slurm':
    os.system("sbatch {}".format(submitfile_full))
elif workload_manager =="condor":
    os.system("condor_submit {}".format(submitfile_full))
elif workload_manager =="bsub":
    os.system("bsub<{}".format(submitfile_full))
time.sleep(3)
