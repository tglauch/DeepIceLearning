#!/usr/bin/env python

import pydag
import os
import argparse
import time
import numpy as np
from configparser import ConfigParser
import fnmatch
import cPickle as pickle

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_config",
        help="dataset config ",
        type=str, default='create_dataset.cfg')
    parser.add_argument(
        "--Rescue",
        help="If true, run in rescue mode ",
        action="store_true")
    parser.add_argument(
        "--filesperJob",
        help="n files per job ", default=-1,
        type=int)
    parser.add_argument(
        "--name",
        help="Name for the Dagman Files",
        type=str, default='create_dataset')
    parser.add_argument(
        "--create_script",
        help="Different script to execute",
        type=str, default='Create_Data_Files.py')
    parser.add_argument(
        "--request_RAM",
        help="amount of RAM in GB",
        type=int, default=4)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print "\n ############################################"
    print "You are running the script with arguments: "
    args = parseArguments()
    for a in args.__dict__:
        print str(a) + ": " + str(args.__dict__[a])
    #main_parser = ConfigParser()
    dataset_parser = ConfigParser()
    #main_parser.read(config_file)
    dataset_parser.read(args.dataset_config)
    print"############################################\n "

Resc = args.__dict__["Rescue"]
PROCESS_DIR = dataset_parser.get("Basics", "dagman_folder")
if not os.path.exists(PROCESS_DIR):
    os.makedirs(PROCESS_DIR)

WORKDIR = os.path.join(PROCESS_DIR, "jobs/")
script = os.path.join(
    "/data/user/mkronmueller/code/DeepIceLearning", 
    #main_parser.get("Basics", "thisfolder"),\
    args.__dict__["create_script"])
dag_name = args.__dict__["name"]
dagFile = os.path.join(
    WORKDIR, "job_{}.dag".format(dag_name))
submitFile = os.path.join(WORKDIR, "job_{}.sub".format(dag_name))

if not Resc:
    if not os.path.exists(WORKDIR):
        os.makedirs(WORKDIR)
        print "Created New Folder in: {}".format(WORKDIR)
    log_path = os.path.join(PROCESS_DIR, "logs/{}/".format(dag_name))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        print "Created New Folder in: {}".format(log_path)
    print "Write Dagman Files to: {}".format(submitFile)
    RAM_str = "{} GB".format(args.__dict__["request_RAM"])
    #####################################################################
    #arguments = " --filelist $(PATH) --dataset_config $(DATASET) "
    arguments = " --filelist $(PATHs) --dataset_config $(DATASET) "
    #####################################################################
    submitFileContent = {"universe": "vanilla",
                         "notification": "Error",
                         "log": "$(LOGFILE).log",
                         "output": "$(LOGFILE).out",
                         "error": "$(LOGFILE).err",
                         "request_memory": RAM_str,
                         "arguments": arguments}
    submitFile = pydag.htcondor.HTCondorSubmit(submitFile,
                                               script,
                                               **submitFileContent)
    submitFile.dump()
    folderlist = dataset_parser.get("Basics", "folder_list")
    #####################################################################
    #basepath = dataset_parser.get("Basics", "MC_path0")
    basepath= []
    for i in xrange(3): # zum testen
    #for i in xrange(len(args['filelist'])):
        a = "MC_path" + str(i)
        basepath.append(str(dataset_parser.get('Basics', a)))
    ###################################################################
    
    filelist = dataset_parser.get("Basics", "file_list")
    filesperjob = args.filesperJob
    file_bunches = []
    #list_file_bunches =[] # unnoetig, da wir davon ausgehen, dass alle file_bunches die gleiche laenge haben

    if folderlist == 'allinmcpath': ##############################################
        folderlist = []
        for i in xrange(len(basepath)):
            a = [subfolder + "/clsim-base-4.0.3.0.99_eff" for subfolder in os.listdir(basepath[i])
                      if os.path.isdir(os.path.join(basepath[i], subfolder))]
            #a =[b + "clsim-base-4.0.3.0.99_eff/"]
            if "logs" in a:
                a.remove("logs") 
            folderlist.append(a)
        #print "\n ############################################"
        #print "Test: "
        #print basepath
        #print folderlist
        #print"############################################\n "
        ############################################################################
    else:
        folderlist = [folder.strip() for folder in folderlist.split(',')]

    if not filelist == 'allinfolder':
        filelist = filelist.split(',')
    run_filelist = []
    
    for j, sim in enumerate(folderlist):
        outfolder = dataset_parser.get('Basics', 'out_folder') +"/filelists/dataset" + str(j)
        #file_bunches = []
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        #lol = 0
        for i, folder in enumerate(folderlist[j]):
            #while lol < 2:
            print folder
            for root, dirs, files in os.walk(os.path.join(basepath[j], folder)):
                i3_files_all = [single_file for single_file in files
                                if single_file[-6:] == 'i3.bz2']
                if not filelist == 'allinfolder':
                    i3_files = []
                    for single_file in i3_files_all:
                        if np.any(
                            [fnmatch.fnmatch(single_file, syntax)
                             for syntax in filelist]):
                            i3_files.append(single_file)
                else:
                    i3_files = i3_files_all
                if len(i3_files) > 0:
                    b = [os.path.join(root, single_file)
                                         for single_file in i3_files]
                    run_filelist.extend(b)
            if filesperjob == -1:
                with open(os.path.join(outfolder,
                                       'File_{}.pickle'.format(i)), 'w+') as f:
                    pickle.dump(run_filelist, f)
                file_bunches.append('File_{}'.format(i))
                run_filelist
        if filesperjob != -1:
            run_filelist = [run_filelist[i:i + filesperjob]
                            if (i + filesperjob) < len(run_filelist)
                            else run_filelist[i:] for i in np.arange(
                                0, len(run_filelist), filesperjob)]
            for i, single_filelist in enumerate(run_filelist):
                with open(os.path.join(outfolder,
                                       'File_{}.pickle'.format(i)), 'w+') as f:
                    pickle.dump(run_filelist[0], f)
                file_bunches.append('File_{}'.format(i))
        #list_file_bunches.append(file_bunches)
    nodes = []
    #for j, file_bunches in enumerate(list_file_bunches):
    for i, bunch in enumerate(file_bunches):
        logfile = log_path + bunch
        PATH = str()
        for k in xrange(len(basepath)):
            PATH = PATH + os.path.join(dataset_parser.get('Basics', 'out_folder') +"/filelists/dataset" + str(k),'{}.pickle'.format(bunch)) +" "
        dagArgs = pydag.dagman.Macros(LOGFILE=logfile,
                                      PATHs=PATH,
                                      #PATH=os.path.join(outfolder,
                                      #                  '{}.pickle'.format(bunch)),
                                      DATASET=args.dataset_config)
        node = pydag.dagman.DAGManNode(i, submitFile)
        node.keywords["VARS"] = dagArgs
        nodes.append(node)
    dag = pydag.dagman.DAGManJob(dagFile, nodes)
    dag.dump()
os.system("condor_submit_dag -f " + dagFile)
time.sleep(1)
