#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse
from configparser import ConfigParser
import datetime
import shutil


def parseArguments():

  parser = argparse.ArgumentParser()
  parser.add_argument("--project", help="The name for the Project", type=str ,default='some_NN')
  parser.add_argument("--input", help="Name of the input files seperated by :", type=str ,default='all')
  parser.add_argument("--model", help="Name of the File containing the model", type=str, default='simple_CNN.cfg')
  parser.add_argument("--virtual_len", help="Use an artifical array length (for debugging only!)", type=int , default=-1)
  parser.add_argument("--continue", help="Give a folder to continue the training of the network", type=str, default = 'None')
  parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')
  args = parser.parse_args()
  return args

def make_condor(request_gpus, request_memory, requirements, addpath, file_location, arguments):
	submit_info = '\
	executable   = ../Neural_Network.py \n \
	universe     = vanilla  \n\
	request_gpus = {0} \n\
	request_memory = {1}GB \n\
	requirements = {2} \n\
	log          = {3}/condor.log \n\
	output       = {3}/condor.out \n\
	error        = {3}/condor.err \n\
	stream_output = True \n\
	getenv = True \n \
	IWD = {4} \n\
	arguments =  {5} \n\
	queue 1 \n\
	'.format(request_gpus, request_memory, requirements, addpath, file_location, arguments)
	return submit_info

def make_slurm(request_gpus, request_memory, addpath, file_location, arguments):

	submit_info = \
'#!/usr/bin/env bash\n\
#SBATCH --time=24:00:00\n\
#SBATCH --partition=gpu\n\
#SBATCH --gres=gpu:{0}\n\
#SBATCH --mem={1} \n\
#SBATCH --error={2}/condor.err\n\
#SBATCH --output={2}/condor.out\n\
#SBATCH --workdir={3}\n\
##SBATCH --exclude=bigbird\n\
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64\n\
python Neural_Network.py {4} \n'.\
format(request_gpus, int(request_memory), addpath, file_location, arguments)
	return submit_info

args = parseArguments().__dict__
parser = ConfigParser()
parser.read('../config.cfg')

file_location = parser.get('Basics', 'thisfolder')
workload_manager = parser.get('Basics', 'workload_manager')
request_gpus = parser.get('GPU', 'request_gpus')
request_memory = parser.get('GPU', 'request_memory')
requirements = parser.get('GPU', 'requirements')
project_name = args['project']

if workload_manager != 'slurm' and workload_manager != 'condor':
	raise NameError('Workload manager not defined. Should either be condor or slurm!')

if args['input'] == 'lowE':
	files = '11029_00000-00999.h5:11029_01000-01999.h5:11029_02000-02999.h5:11029_03000-03999.h5:11029_04000-04999.h5:11029_05000-05999.h5'
elif args['input'] == 'highE':
	files = '11069_00000-00999.h5:11069_01000-01999.h5:11069_02000-02999.h5:11069_03000-03999.h5:11069_04000-04999.h5:11069_05000-05999.h5'
else:
	files = args['input']


if args['continue'] != 'None':
	arguments = '--continue {}'.format(args['continue'])
	addpath = args['continue']
	if addpath[-1]=='/':
		addpath= addpath[:-1]

	if workload_manager == 'slurm':
		submit_info = make_slurm(request_gpus, float(request_memory)*1e3, addpath, file_location, arguments)
	elif workload_manager == 'condor':
		submit_info = make_condor(request_gpus, request_memory, requirements, addpath, file_location, arguments)

else:
	today = str(datetime.date.today())
	folders=['train_hist/',
	 'train_hist/{}'.format(today),
	 'train_hist/{}/{}'.format(today, project_name)]
	for folder in folders:
	    if not os.path.exists('{}'.format(os.path.join(file_location,folder))):
	        os.makedirs('{}'.format(os.path.join(file_location,folder)))

	arguments = ''
	for a in args:
	  if not a == 'input':
	  	arguments += '--{} {} '.format(a, args[a])
	  else:
		arguments += '--input {} '.format(files)  	

	print("\n --------- \n \
		You are running the script with arguments: \n {}  \
		\n --------- \n").format(arguments)

	arguments += '--date {} '.format(today)
	arguments += '--ngpus {} '.format(request_gpus)
	addpath = os.path.join('train_hist',today,project_name)

	if workload_manager == 'slurm':
		submit_info = make_slurm(request_gpus, float(request_memory)*1e3, addpath, file_location, arguments)

	elif workload_manager == 'condor':
		submit_info = make_condor(request_gpus, request_memory, requirements, addpath, file_location, arguments)

print(submit_info) 

with open('../submit.sub', 'w') as file:
	file.write(submit_info)

os.chdir('../')

if workload_manager == 'slurm':
	os.system("sbatch submit.sub")
else:
	os.system("condor_submit submit.sub")

time.sleep(3)

if os.path.exists(os.path.join(file_location, addpath,'submit.sub')):
	os.remove(os.path.join(file_location, addpath,'submit.sub'))

shutil.move('submit.sub', os.path.join(file_location, addpath))
