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
    parser.add_argument("--dataset_config", help="dataset config ",\
                        type=str ,default='create_dataset.cfg')
    parser.add_argument("--Rescue", help="If true, run in rescue mode ", action="store_true")
    parser.add_argument("--filesperJob", help="n files per job ", default=-1,\
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

WORKDIR = os.path.join(PROCESS_DIR, "jobs/")
script = os.path.join(main_parser.get("Basics","thisfolder"),"Create_Data_Files.py")
dag_name = args.__dict__["name"]
dagFile = os.path.join(WORKDIR,"job_{}.dag".format(dag_name))
submitFile = os.path.join(WORKDIR, "job_{}.sub".format(dag_name))

if not Resc:
    if not os.path.exists(WORKDIR):
        os.makedirs(WORKDIR)
        print "Created New Folder in: "+ WORKDIR
    log_path=os.path.join(PROCESS_DIR,"logs/{}/".format(dag_name))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        print "Created New Folder in: "+ log_path
    print "Write Dagman Files to: "+submitFile
    arguments = " --filelist $(PATH) --dataset_config $(DATASET) "
    submitFileContent = {"universe": "vanilla",
                      "notification": "Error",
                      "log": "$(LOGFILE).log",
                      "output": "$(LOGFILE).out",
                      "error": "$(LOGFILE).err",
                      "request_memory": "1.5GB",
                     # "IWD" : main_parser.get("Basics","thisfolder"),
                      "arguments": arguments}
    submitFile = pydag.htcondor.HTCondorSubmit(submitFile,
                                               script,
                                               **submitFileContent)
    submitFile.dump()


    folderlist = dataset_parser.get("Basics","folder_list")
    basepath = dataset_parser.get("Basics","MC_path") 
    filelist = dataset_parser.get("Basics","file_list")
    filesperjob = args.filesperJob
    outfolder = dataset_parser.get('Basics', 'out_folder')
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    file_bunches= []
    
    if folderlist == 'allinmcpath':
        folderlist =  [subfolder for subfolder in os.listdir(basepath) \
                       if os.path.isdir(os.path.join(basepath,subfolder))]
    else:
        folderlist = [folder.strip() for folder in folderlist.split(',')]

    if not filelist == 'allinfolder':
        filelist = filelist.split(',')
    run_filelist = []
    for i, folder in enumerate(folderlist):
        print folder
        for root, dirs, files in os.walk(os.path.join(basepath, folder)):
            i3_files_all = [single_file for single_file in files if single_file[-6:]=='i3.bz2']
            if not filelist == 'allinfolder':
                i3_files = []
                for single_file in i3_files_all:
                    if np.any([fnmatch.fnmatch(single_file, syntax) for syntax in filelist]):
                        i3_files.append(single_file)
            else:
                i3_files = i3_files_all
            if len(i3_files)>0:
                run_filelist.extend([os.path.join(root, single_file) for single_file in i3_files])
        if filesperjob == -1:
            with open(os.path.join(outfolder, 'File_{}.txt'.format(i)), 'w+') as f:
                for line in run_filelist:
                    f.write("%s \n" % line)
            file_bunches.append('File_{}'.format(i))
            run_filelist = []
    if filesperjob != -1 :
        run_filelist = [run_filelist[i:i+filesperjob] if (i+filesperjob) < len(run_filelist)\
                        else run_filelist[i:] for i in np.arange(0, len(run_filelist), filesperjob)]
        for i, single_filelist in enumerate(run_filelist):
            with open(os.path.join(outfolder, 'File_{}.txt'.format(i)), 'w+') as f:
                for line in run_filelist:
                    f.write("%s \n" % line) 
            file_bunches.append('File_{}'.format(i))
    nodes  = []
    for i, bunch in enumerate(file_bunches):
        logfile = log_path+bunch
        dagArgs = pydag.dagman.Macros(LOGFILE=logfile,
                                      PATH = os.path.join(outfolder, '{}.txt'.format(bunch)),
                                      DATASET = args.dataset_config)
        node = pydag.dagman.DAGManNode(i, submitFile)
        node.keywords["VARS"] = dagArgs
        nodes.append(node)

    dag = pydag.dagman.DAGManJob(dagFile, nodes)
    dag.dump()
os.system("condor_submit_dag -f "+dagFile)
time.sleep(1)
