#!/usr/bin/env python

import pydag
import datetime
import os
import argparse
import time
import numpy as np
from configparser import ConfigParser
import cPickle as pickle


def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_config",
        help="dataset config ",
        type=str, default='create_dataset.cfg')
    parser.add_argument(
        "--filesperJob",
        help="n files per job ", default=50,
        type=int)
    parser.add_argument(
        "--name",
        help="Name for the Dagman Files",
        type=str, default='create_dataset')
    parser.add_argument(
        "--request_RAM",
        help="amount of RAM in GB",
        type=int, default=4)
    parser.add_argument(
        "--rescue",
        help="Run rescue script?!",
        type=str, default='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print "\n ############################################"
    print "You are running the script with arguments: "
    args = parseArguments()
    for a in args.__dict__:
        print str(a) + ": " + str(args.__dict__[a])
    dataset_parser = ConfigParser()
    dataset_parser.read(args.dataset_config)
    print"############################################\n "

Resc = args.__dict__["rescue"]
if Resc == '':
    today = str(datetime.datetime.now()).\
        replace(" ", "-").split(".")[0].replace(":", "-")

    PROCESS_DIR = os.path.join(dataset_parser.get("Basics", "dagman_folder"),
                               today)
    if not os.path.exists(PROCESS_DIR):
        os.makedirs(PROCESS_DIR)

    WORKDIR = os.path.join(PROCESS_DIR, "jobs/")
    script = os.path.join(
        dataset_parser.get("Basics", "thisfolder"),
        'submit_scripts/create_dataset_env.sh')
    print('Submit Script:\n {}'.format(script))

    dag_name = args.__dict__["name"]
    dagFile = os.path.join(
        WORKDIR, "job_{}.dag".format(dag_name))
    submitFile = os.path.join(WORKDIR, "job_{}.sub".format(dag_name))
    if not os.path.exists(WORKDIR):
        os.makedirs(WORKDIR)
        print("Created New Folder in: {}".format(WORKDIR))

    log_path = os.path.join(PROCESS_DIR, "logs/{}/".format(dag_name))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        print("Created New Folder in: {}".format(log_path))

    print("Write Dagman Files to: {}".format(submitFile))
    RAM_str = "{} GB".format(args.__dict__["request_RAM"])
    arguments = " --filelist $(PATHs) --dataset_config $(DATASET) "
    submitFileContent = {"universe": "vanilla",
                         "notification": "Error",
                         "log": "$(LOGFILE).log",
                         "output": "$(LOGFILE).out",
                         "error": "$(LOGFILE).err",
	#		             "Requirements" : "HAS_CVMFS_icecube_opensciencegrid_org",
    #                    "Requirements" : '(Machine != "n-15.icecube.wisc.edu")',
                         "request_memory": RAM_str,
                         "arguments": arguments}
    submitFile = pydag.htcondor.HTCondorSubmit(submitFile,
                                               script,
                                               **submitFileContent)
    submitFile.dump()
    folderlist = dataset_parser.get("Basics", "folder_list")
    basepath = [dataset_parser['Basics'][key] for key in
                dataset_parser['Basics'].keys() if 'mc_path' in key]
    filelist = dataset_parser.get("Basics", "file_list")
    filesjob = args.filesperJob
    file_bunches = []

    if folderlist == 'allinmcpath':
        folderlist = []
        for p in basepath:
            a = [subfolder for subfolder in os.listdir(p)
                 if os.path.isdir(os.path.join(p, subfolder))]
            if "logs" in a:
                a.remove("logs")
            if "job_stats.json.gz" in a:
                a.remove("job_stats.json.gz")
            folderlist.append(a)
    else:
        folderlist = [folder.strip() for folder in folderlist.split(',')]

    if not filelist == 'allinfolder':
        filelist = filelist.split(',')

    for j, sim in enumerate(folderlist):
        outfolder = os.path.join(dataset_parser.get('Basics', 'out_folder'),
                                 "filelists/dataset",
                                 str(j))
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        run_filelist = []
        for ii, folder in enumerate(folderlist[j]):
            for root, dirs, files in os.walk(os.path.join(basepath[j],
                                                          folder)):
                i3_files_all = [s_file for s_file in files
                                if s_file[-6:] == 'i3.bz2']
                if not filelist == 'allinfolder':
                    i3_files = [f for f in filelist if f in i3_files_all]
                else:
                    i3_files = i3_files_all
                if len(i3_files) > 0:
                    b = [os.path.join(root, single_file)
                         for single_file in i3_files]
                    run_filelist.extend(b)
            run_filelist = [run_filelist[i:i + filesjob] for i
                            in np.arange(0, len(run_filelist), filesjob + 1)]
            for numberInRunFilelist, single_filelist in enumerate(run_filelist):
                with open(os.path.join(outfolder,
                                       'File_{}.pickle'.format(ii * 20 + numberInRunFilelist)), 'w+') as f:
                    pickle.dump(single_filelist, f)
            run_filelist = []


#################### if Job number is set by hand ################
    misty = 0
    while misty <= 999:  # bad quick fix ###### AAAAAAAAAAAHHHH
        file_bunches.append('File_{}'.format(misty))
        misty += 1
###################################################################

    nodes = []
    for i, bunch in enumerate(file_bunches):
        logfile = log_path + bunch
        PATH = ''
        for k in xrange(len(basepath)):
            PATH = PATH +\
                os.path.join(dataset_parser.get('Basics', 'out_folder'),
                             "filelists/dataset",
                             str(k),
                             '{}.pickle'.format(bunch)) + " "
        dagArgs = pydag.dagman.Macros(LOGFILE=logfile,
                                      PATHs=PATH,
                                      DATASET=args.dataset_config)
        node = pydag.dagman.DAGManNode(i, submitFile)
        node.keywords["VARS"] = dagArgs
        nodes.append(node)
    dag = pydag.dagman.DAGManJob(dagFile, nodes)
    dag.dump()
else:
    dagFile = Resc
os.system("condor_submit_dag -f " + dagFile)
time.sleep(1)
