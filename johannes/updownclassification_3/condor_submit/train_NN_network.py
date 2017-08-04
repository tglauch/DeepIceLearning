#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse
from ConfigParser import ConfigParser
import datetime
import shutil
import subprocess
import itertools


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", help="The name for the Project", type=str ,default='some_NN')
    parser.add_argument("--input", help="Name of the input files seperated by :", type=str ,default='all')
    parser.add_argument("--using", help="charge or time", type=str, default='time')
    parser.add_argument("--model", help="Name of the File containing the model", type=str, default='simple_FCNN.cfg')
    parser.add_argument("--virtual_len", help="Use an artifical array length (for debugging only!)", type=int , default=-1)
    parser.add_argument("--continue", help="Give a folder to continue the training of the network", type=str, default = 'None')
    parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')
    args = parser.parse_args()
    return args


args = parseArguments().__dict__

parser = ConfigParser()
parser.read('../config.cfg')
file_location = parser.get('Basics', 'thisfolder')
request_gpus = parser.get('GPU', 'request_gpus')
request_memory = parser.get('GPU', 'request_memory')
requirements = parser.get('GPU', 'requirements')
project_name = args['project']

low_files = ['11029_00000-00999.h5','11029_01000-01999.h5','11029_02000-02999.h5','11029_03000-03999.h5','11029_04000-04999.h5','11029_05000-05999.h5']
high_files = ['11069_00000-00999.h5','11069_01000-01999.h5','11069_02000-02999.h5','11069_03000-03999.h5','11069_04000-04999.h5','11069_05000-05999.h5','11069_06000-06999.h5']

if args['input'] == 'all':
    files = ':'.join(low_files + high_files)
elif args['input'] == 'lowE':
	files = ':'.join(low_files)
elif args['input'] == 'highE':
	files = ':'.join(high_files)
elif args['input'] == 'highE_reduced': #same as highE, but only three files
    files = ':'.join(high_files[0:3])
elif len(args['input']) > 0 and (args['input'][0] == 'h' or args['input'][0] == 'l') and args['input'][1].isdigit():
    inputs = {k:v for k, v in zip(*[iter(["".join(x) for _, x in itertools.groupby(args['input'], key=str.isdigit)])]*2)}
    files = ''
    if 'h' in inputs:
        files = ':'.join([high_files[i] for i in map(int, list(inputs['h']))])
    if 'l' in inputs:
        files = ':'.join(filter(lambda s: len(s) > 0, files.split(':')) + [low_files[i] for i in map(int, list(inputs['l']))])
else:
	files = args['input']


if args['continue'] != 'None':
	arguments = '--continue {}'.format(args['continue'])

	addpath = args['continue']
	if addpath[-1]=='/': # remove slash at the end
		addpath= addpath[:-1]

	submit_info = '\
	executable   = ../updown_network.py \n \
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

else:
	today = str(datetime.date.today())
	folders=['train_hist/',
	 'train_hist/{}'.format(today),
	 'train_hist/{}/{}'.format(today, project_name)]

	for folder in folders:
	    if not os.path.exists('{}'.format(os.path.join(file_location,folder))):
	        os.makedirs('{}'.format(os.path.join(file_location,folder)))

	print("\n ---------")
	print("You are running the submit script with arguments: ")
	arguments = ''
	for a in args:
		print('{} : {}'.format(a, args[a]))
		if not a == 'input':
			arguments += '--{} {} '.format(a, args[a])
		else:
			arguments += '--input {} '.format(files)
	print("--------- \n")

	arguments += '--date {}'.format(today)
	addpath = os.path.join('train_hist',today,project_name)
	submit_info = '\
	executable   = ../updown_network.py \n \
	universe     = vanilla  \n\
	request_gpus = {0} \n\
	request_memory = {1}GB \n\
	requirements = {2} \n\
	log          = {3}/condor.log \n\
	output       = {3}/condor.out \n\
	error        = {3}/condor.err \n\
	getenv = True \n \
	stream_output = True \n\
	IWD = {4} \n\
	arguments =  {5} \n\
	queue 1 \n\
	'.format(request_gpus, request_memory, requirements, addpath, file_location, arguments)

print(submit_info)

with open('submit.sub', 'w') as f:
	f.write(submit_info)

#os.system("condor_submit submit.sub")
output = subprocess.check_output("condor_submit submit.sub", shell=True)
print output
time.sleep(3)
with open('submit.sub', 'a') as f:
    f.write(output)
shutil.copy('submit.sub', os.path.join(file_location, addpath))
