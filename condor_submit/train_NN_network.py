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


args = parseArguments().__dict__

parser = ConfigParser()
parser.read('../config.cfg')
file_location = parser.get('Basics', 'thisfolder')
request_gpus = parser.get('GPU', 'request_gpus')
request_memory = parser.get('GPU', 'request_memory')
requirements = parser.get('GPU', 'requirements')
project_name = args['project']


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

else:
	today = str(datetime.date.today())
	folders=['train_hist/',
	 'train_hist/{}'.format(today),
	 'train_hist/{}/{}'.format(today, project_name)]

	for folder in folders:
	    if not os.path.exists('{}'.format(os.path.join(file_location,folder))):
	        os.makedirs('{}'.format(os.path.join(file_location,folder)))

	print("\n ---------")
	print("You are running the script with arguments: ")
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
	executable   = ../Neural_Network.py \n \
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

with open('submit.sub', 'w') as file:
	file.write(submit_info)

os.system("condor_submit submit.sub")
time.sleep(3)
shutil.copy('submit.sub', os.path.join(file_location, addpath))
