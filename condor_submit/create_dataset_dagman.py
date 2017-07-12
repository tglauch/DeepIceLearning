#!/usr/bin/env python

import pydag
import itertools
import sys
import os
import argparse
import time
import numpy as np

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--Rescue", help="If true, run in rescue mode ", action="store_true")
    parser.add_argument("--name", help="Name for the Dagman Files", type=str ,default='create_dataset')
    args = parser.parse_args()

    return args


args=parseArguments()

if __name__ == '__main__':
	print"\n ############################################"
	print("You are running the script with arguments: ")
	for a in args.__dict__:
	    print(str(a) + ": " + str(args.__dict__[a]))
	print"############################################\n "

Resc=args.__dict__["Rescue"]

PROCESS_DIR = "/data/user/tglauch/ML_Reco/condor_submit/create_dataset_dagman/" #'/scratch/tglauch/dagman_files/' 
WORKDIR = PROCESS_DIR+"jobs/"
script = "/data/user/tglauch/ML_Reco/Create_Data_Files.py"
dag_name = args.__dict__["name"]
dagFile = WORKDIR+"job_{}.dag".format(dag_name)
submitFile = WORKDIR+"job_{}.sub".format(dag_name)

if not Resc:
	if not os.path.exists(WORKDIR):
		os.makedirs(WORKDIR)
		print "Created New Folder in:  "+ WORKDIR
	
	path=PROCESS_DIR+"logs/{}/".format(dag_name)
	
	if not os.path.exists(path):
			os.makedirs(path)
			print "Created New Folder in:  "+ path
	print "Write Dagman Files to: "+submitFile
	
	arguments = " --project $(ARG1) --num_files $(ARG2) --folder $(ARG3)"
	print arguments
	submitFileContent = {"universe": "vanilla",
	                     "notification": "Error",
	                     "log": "$(LOGFILE).log",
	                     "output": "$(LOGFILE).out",
	                     "error": "$(LOGFILE).err",
	                     "request_memory": "1.5GB",
	                     "arguments": arguments}
	
	submitFile = pydag.htcondor.HTCondorSubmit(submitFile,
	                                           script,
	                                           **submitFileContent)
	submitFile.dump()
	args = np.genfromtxt('folderlist.txt', dtype = object)
	nodes  = []
	for i, arg in enumerate(args): #for i, a1, a2 in enumerate(itertools.product(args1, args2))
	    logfile = path+str(arg[1].replace('/','_').replace(':','_'))
	    dagArgs = pydag.dagman.Macros(
	                LOGFILE=logfile,
			ARG1 = arg[2],
			ARG2 = arg[0],
			ARG3 = arg[1])
	    node = pydag.dagman.DAGManNode(i, submitFile)
	    node.keywords["VARS"] = dagArgs
	    nodes.append(node)
	
	dag = pydag.dagman.DAGManJob(dagFile, nodes)
	dag.dump()

os.system("condor_submit_dag -f "+dagFile)
time.sleep(1)
