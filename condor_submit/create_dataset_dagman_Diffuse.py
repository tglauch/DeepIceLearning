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
    parser.add_argument("--filesperJob", help="n files per job ", default=5,\
                        type=int)
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
script = main_parser.get("Basics","thisfolder")+"/Create_Data_Files_Diffuse.py"
dag_name = args.__dict__["name"]
dagFile = WORKDIR+"job_{}.dag".format(dag_name)
submitFile = WORKDIR+"job_{}.sub".format(dag_name)

if not Resc:
    if not os.path.exists(WORKDIR):
        os.makedirs(WORKDIR)
        print "Created New Folder in: "+ WORKDIR
    log_path=PROCESS_DIR+"/logs/{}/".format(dag_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        print "Created New Folder in: "+ log_path
    print "Write Dagman Files to: "+submitFile
    arguments = " --project $(PROJECT) --files $(FILES) --main_config $(MAIN)"\
            +" --dataset_config $(DATASET) "
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

    file_list = dataset_parser.get("Basics","file_list")
    if file_list=="allinfolder":
        files = []
        for f_name in os.listdir(basepath+"/"+mc_folder):
            if not f_name[-6:]=="i3.bz2":
                continue
            cur_f_id = f_name.split(mc_folder.strip("/")+".")[1].strip(".i3.bz2")
            files.append(cur_f_id)
    elif "-" in file_list:
        files = []
        low, up = file_list.split("-")
        for f_id in range(int(low), int(up)):
            files.append('{0:06d}'.format(f_id))
    elif ":" in file_list:
        files = file_list.split(":")
    else:
        print "file_list in dataset_config not understood. Have a look at the\
        options there!"
    nodes  = []
    filesperJob = args.filesperJob*1. if len(files)>args.filesperJob else 1.
    file_bunches = list(np.array_split(files,
                                       int(len(files)/filesperJob)))
    for i, bunch in enumerate(file_bunches): #for i, a1, a2 in enumerate(itertools.product(args1, args2))
        logfile = log_path+bunch[0]
        bunch_str = " "
        for s in bunch:
            bunch_str +=" {}".format(s)
        print "files in this job: ", bunch_str
        dagArgs = pydag.dagman.Macros(LOGFILE=logfile,
                                      PROJECT = args.project,
                                      FILES = bunch_str,
                                      MAIN = args.main_config,
                                      DATASET = args.dataset_config)
        node = pydag.dagman.DAGManNode(i, submitFile)
        node.keywords["VARS"] = dagArgs
        nodes.append(node)

    dag = pydag.dagman.DAGManJob(dagFile, nodes)
    dag.dump()
os.system("condor_submit_dag -f "+dagFile)
time.sleep(1)
