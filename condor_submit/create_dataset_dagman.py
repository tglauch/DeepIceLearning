#!/usr/bin/env python

import pydag
import itertools
import sys
import os
import argparse
import time
import numpy as np
from configparser import ConfigParser

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_config", help="main config file",\
                        type=str ,default='default.cfg')
    parser.add_argument("--dataset_config", help="dataet config ",\
                        type=str ,default='create_dataset.cfg')
    parser.add_argument("--project", help="The name for the Project",\
                        type=str ,default='some_NN')
    parser.add_argument("--Rescue", help="If true, run in rescue mode ", action="store_true")
    parser.add_argument("--name", help="Name for the Dagman Files", type=str ,default='create_dataset')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print "\n ############################################"
    print "You are running the script with arguments: "
    args = parseArguments()
    for a in args.__dict__:
        print str(a) + ": " + str(args.__dict__[a])
    config_file = args.main_config
    main_parser = ConfigParser()
    dataset_parser = ConfigParser()
    main_parser.read(config_file)
    dataset_parser.read(args.dataset_config)

    print"############################################\n "

Resc=args.__dict__["Rescue"]

PROCESS_DIR = dataset_parser.get("Basics","dagman_folder")
if not os.path.exists(PROCESS_DIR):
    os.makedirs(PROCESS_DIR)

WORKDIR = PROCESS_DIR+"/jobs/"
script = main_parser.get("Basics","thisfolder")+"/Create_Data_Files.py"
dag_name = args.__dict__["name"]
dagFile = WORKDIR+"job_{}.dag".format(dag_name)
submitFile = WORKDIR+"job_{}.sub".format(dag_name)

if not Resc:
    if not os.path.exists(WORKDIR):
        os.makedirs(WORKDIR)
        print "Created New Folder in: "+ WORKDIR
    path=PROCESS_DIR+"/logs/{}/".format(dag_name)
    if not os.path.exists(path):
        os.makedirs(path)
        print "Created New Folder in: "+ path
    print "Write Dagman Files to: "+submitFile
    arguments = " --project $(PROJECT) --folder $(FOLDER) --main_config $(MAIN)"\
            +" --dataset_config $(DATASET) "

    ##--num-files = -1 ? 
    submitFileContent = {"universe": "vanilla",
                      "notification": "Error",
                      "log": "$(LOGFILE).log",
                      "output": "$(LOGFILE).out",
                      "error": "$(LOGFILE).err",
                      "request_memory": "1.5GB",
                      "IWD" : main_parser.get("Basics","thisfolder"),
                      "arguments": arguments}
    submitFile = pydag.htcondor.HTCondorSubmit(submitFile,
                                               script,
                                               **submitFileContent)
    submitFile.dump()
    file_list = dataset_parser.get("Basics","file_list").split(":")
    print "files to be read: " , file_list
    nodes  = []
    for i, filename in enumerate(file_list): #for i, a1, a2 in enumerate(itertools.product(args1, args2))
        logfile = path+filename.replace('/','_').replace(':','_').strip(".h5")\
                +"_"+str(i)
        dagArgs = pydag.dagman.Macros(LOGFILE=logfile,
                                  PROJECT = args.project,
                                  FOLDER = filename,
                                  MAIN = args.main_config,
                                  DATASET = args.dataset_config)
        node = pydag.dagman.DAGManNode(i, submitFile)
        node.keywords["VARS"] = dagArgs
        nodes.append(node)

    dag = pydag.dagman.DAGManJob(dagFile, nodes)
    dag.dump()
os.system("condor_submit_dag -f "+dagFile)
time.sleep(1)
