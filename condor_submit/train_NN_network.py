#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse
from configparser import ConfigParser
import datetime


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
    args = parser.parse_args()
    return args


def make_condor(request_gpus, request_memory, requirements, addpath,
                arguments, thisfolder):
    submit_info = '\
            executable   = {folder}/Neural_Network.py \n\
            universe     = vanilla  \n\
            request_gpus = {gpu} \n\
            request_memory = {mem}GB \n\
            requirements = {req} \n\
            log          = {addp}/condor.log \n\
            output       = {addp}/condor.out \n\
            error        = {addp}/condor.err \n\
            stream_output = True \n\
            getenv = True \n\
            IWD = {folder} \n\
            arguments =  {args} \n\
            queue 1 \n '.format(gpu=request_gpus, mem=request_memory,
                                req=requirements, addp=addpath,
                                args=arguments, folder=thisfolder)
    return submit_info


def make_slurm(request_gpus, request_memory, condor_folder, file_location,
               arguments, thisfolder, exclude=''):

    if exclude != '':
        exclude_node = '#SBATCH --exclude {} \n'.format(exclude)

    submit_info = '#!/usr/bin/env bash\n\
#SBATCH --time=48:00:00\n\
#SBATCH --partition=gpu\n\
#SBATCH --gres=gpu:{0}\n\
#SBATCH --mem={1} \n\
#SBATCH --error={2}/condor.err\n\
#SBATCH --output={2}/condor.out\n\
{5}\
\n\
python {4}/Neural_Network.py {3} \n'.format(
        request_gpus, int(request_memory),
        condor_folder, arguments, thisfolder, exclude_node)

    return submit_info


args = parseArguments().__dict__
parser = ConfigParser()
if args['continue'] != 'None':
    parser.read(os.path.join(args["continue"], 'config.cfg'))
else:
    parser.read(args["main_config"])

train_location = parser.get('Basics', 'train_folder')
workload_manager = parser.get('Basics', 'workload_manager')
request_gpus = parser.get('GPU', 'request_gpus')
request_memory = parser.get('GPU', 'request_memory')
requirements = parser.get('GPU', 'requirements')
project_name = args['project']
thisfolder = parser.get("Basics", "thisfolder")
if 'exclude_node' in parser['GPU'].keys():
    exclude = parser.get('GPU', 'exclude_node')
else:
    exclude = ' '

if workload_manager != 'slurm' and workload_manager != 'condor':
    raise NameError(
        'Workload manager not defined. Should either be condor or slurm!')

files = args['input']

arguments = ''
for a in args:
    if not a == 'input':
        arguments += '--{} {} '.format(a, args[a])
    else:
        arguments += '--input {} '.format(files)

if args['continue'] != 'None':
    arguments += '--continue {}'.format(args['continue'])
    save_path = args['continue']
    condor_out_folder = os.path.join(args['continue'], 'condor')
    args["model"] = os.path.join(args['continue'], 'model.cfg')
else:
    today = str(datetime.datetime.now()).\
        replace(" ", "-").split(".")[0].replace(":", "-")

    save_path = os.path.join(
        train_location, '{}/{}'.format(project_name, today))
    condor_out_folder = os.path.join(save_path, 'condor')

    if not os.path.exists(condor_out_folder):
        print('Create Condor-Out Folder: \n {}'.format(condor_out_folder))
        os.makedirs(condor_out_folder)
    arguments += ' --save_folder {} '.format(save_path)

arguments += ' --ngpus {} '.format(request_gpus)
if workload_manager == 'slurm':
    submit_info = make_slurm(request_gpus, float(request_memory) * 1e3,
                             condor_out_folder, train_location, arguments,
                             thisfolder, exclude)
elif workload_manager == 'condor':
    submit_info = make_condor(
        request_gpus, request_memory, requirements,
        condor_out_folder, arguments, thisfolder)

print(submit_info)
submitfile_full = os.path.join(condor_out_folder, 'submit.sub')
with open(submitfile_full, "wc") as file:
    file.write(submit_info)

if not os.path.exists(os.path.join(save_path, 'config.cfg')):
    os.system("cp {} {} ".format(
        args["main_config"], os.path.join(save_path, 'config.cfg')))
if not os.path.exists(os.path.join(save_path, 'model.cfg')):
    os.system("cp {} {} ".format(
        args["model"], os.path.join(save_path, 'model.cfg')))

if workload_manager == 'slurm':
    os.system("sbatch {}".format(submitfile_full))
else:
    os.system("condor_submit {}".format(submitfile_full))

time.sleep(3)
