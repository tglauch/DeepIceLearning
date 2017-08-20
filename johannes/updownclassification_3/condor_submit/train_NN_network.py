#!/usr/bin/env python
# coding: utf-8

import os, sys
import time
import argparse
from ConfigParser import ConfigParser
import datetime
import shutil
import subprocess


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", help="The name for the Project", type=str ,default='some_NN')
    parser.add_argument("--continue", help="Give a folder to continue the training of the network", type=str, default = 'None')
    parser.add_argument("--nosubmit", help="If this option is passed, the updownscript will be called directy on cobalt and no \
                        submitter will be used. Forwards all other arguments.", nargs='?', const=True, default=argparse.SUPPRESS)
    parser.add_argument("--version", action="version", version='%(prog)s - Version 3.0')
    args = parser.parse_known_args()
    return args[0], args[1]

parsed = parseArguments()
args = parsed[0].__dict__
unknown = parsed[1]

if 'nosubmit' in args and args['nosubmit']:
	print("\n ---------")
	print("You are running the submit script with arguments: ")
	arguments = ''
	for a in args:
		print('{} : {}'.format(a, args[a]))
		arguments += '--{} {} '.format(a, args[a])
	print "Additional (unknown) arguments are:"
	print ' '.join(unknown)
	print "They will all be passed to the  updown script. No submitter used! (directly called)"
	print("--------- \n")
    
	arguments = []
	for a in filter(lambda s: 'nosubmit' not in s, args):
		arguments.extend(["--" + a, str(args[a])]) # does not work with bool arguments! (convert them to string)
	arguments.extend(unknown)
	print "Passing these arguments:"
	print arguments
	print "Running script and waiting for it to finish. Output will then be printed."
	process = subprocess.Popen(['../updown_network.py'] + arguments,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
	returncode = process.wait()
	#print('script returned {0}'.format(returncode))
	print(process.stdout.read())
    
	sys.exit(0)
    

parser = ConfigParser()
parser.read('../config.cfg')
file_location = parser.get('Basics', 'thisfolder')
request_gpus = parser.get('GPU', 'request_gpus')
request_memory = parser.get('GPU', 'request_memory')
requirements = parser.get('GPU', 'requirements')
project_name = args['project']


if args['continue'] != 'None':
	arguments = '--continue {} '.format(args['continue'])
	arguments = arguments + ' '.join(unknown)

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
		arguments += '--{} {} '.format(a, args[a])
	print "Additional (unknown) arguments are:"
	print ' '.join(unknown)
	print "They will all be passed to the  updown script."
	print("--------- \n")

	arguments += ' '.join(unknown) + ' '
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
#time.sleep(3)
#this would print something like:
#	Submitting job(s).
#	1 job(s) submitted to cluster 263196320.
#something similar is already written to condor.log, so it is not needed anymore. (job id is necessary for killing a job or being #able to understand if it still runs)
#with open('submit.sub', 'a') as f:
#    f.write(output)
shutil.copy('submit.sub', os.path.join(file_location, addpath))
