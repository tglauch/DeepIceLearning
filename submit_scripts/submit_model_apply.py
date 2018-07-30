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
        help="main config file, user-specific",
        type=str, default='default.cfg')
    parser.add_argument(
        "--folder",
        type=str, required=True,
        help="trainfolder ")
    args = parser.parse_args()
    return args



args = parseArguments().__dict__
parser = configparser.ConfigParser()
parser.read(args["main_config"])
parser_dict = {s:dict(parser.items(s)) for s in parser.sections()}

train_location = parser.get('Basics', 'train_folder')
workload_manager = parser.get('Basics', 'workload_manager')
request_gpus = parser.get('GPU', 'request_gpus')
request_memory = parser.get('GPU', 'request_memory')
requirements = parser.get('GPU', 'requirements')
thisfolder = parser.get("Basics", "thisfolder")
if 'exclude_node' in parser_dict['GPU'].keys():
    exclude = parser.get('GPU', 'exclude_node')
else:
    exclude = ' '

if workload_manager not in ['slurm','condor','bsub']:
    raise NameError(
        'Workload manager not defined!')

#load shelf:
#shelf = shelve.open(os.path.join(args['folder'],'run_info.shlf'))

arguments = ' --main_config {}  --folder {}'.format(\
                        args['main_config'], args['folder'])

save_path = args['folder']
condor_out_folder = os.path.join(save_path, 'condor')
if workload_manager == 'slurm':
    submit_info = make_slurm("Apply_Model.py",\
                             request_gpus,\
                             float(request_memory) * 1e3,\
                             condor_out_folder,\
                             train_location,\
                             arguments,\
                             thisfolder,\
                             exclude)
elif workload_manager == 'condor':
    submit_info = make_condor("Apply_Model.py",\
                              request_gpus,\
                              request_memory,\
                              requirements,\
                              condor_out_folder,\
                              arguments,\
                              thisfolder)
elif workload_manager == 'bsub':
    submit_info = make_bsub("Apply_Model.py",\
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
