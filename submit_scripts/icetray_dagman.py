#!/usr/bin/env python

import pydag
import datetime
import os
import argparse
import time
import numpy as np
from configparser import ConfigParser
import math
from functions import get_files_from_folder, harvest_generators
import shutil
import pickle

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_config",
        help="dataset config ",
        type=str, default='create_dataset.cfg')
    parser.add_argument(
        "--files_per_job",
        help="n files per job ", default=50,
        type=int)
    parser.add_argument(
        "--name",
        help="Name for the Dagman Files",
        type=str, default='process_classification')
    parser.add_argument(
        "--request_RAM",
        help="amount of RAM in GB",
        type=int, default=4)
    parser.add_argument(
        "--compression_format",
        help="which compression format to use",
        type=str, default='i3.bz2', nargs='+')
    parser.add_argument(
        "--must_contain",
        help="strings that must be in filename",
        type=str, nargs='+')
    parser.add_argument(
        "--exclude",
        help="strings that must be in filename",
        type=str, nargs='+')
    parser.add_argument(
        "--rescue",
        help="Run rescue script?!",
        type=str, default='')
    parser.add_argument(
        "--files_per_dataset",
        help="number of files per dataset",
        type=int)
    parser.add_argument(
        "--gcd",
        help="path to gcd file",
        type=str)
    parser.add_argument(
        "--muongun",
        help="If create generator with this savepath",
        type=str)
    parser.add_argument(
        "--gpu", action='store_true', default=False)
    args = parser.parse_args()
    return args.__dict__


if __name__ == '__main__':
    print "\n ############################################"
    print "You are running the script with arguments: "
    args = parseArguments()
    for a in args:
        print str(a) + ": " + str(args[a])
    print"############################################\n "
    dataset_parser = ConfigParser()
    dataset_parser.read(args['dataset_config'])
    today = str(datetime.datetime.now()).\
        replace(" ", "-").split(".")[0].replace(":", "-")
    rescue_file = args["rescue"]
    outfolder = dataset_parser.get('Basics', 'out_folder')
    if rescue_file != '':
        good_jobs = []
        with open(rescue_file, "r") as inf:
            content = inf.readlines()
            good_jobs = []
            for l in content:
                if (not "#" in l) and len(l)>1:
                    words = l.split(" ")
                    good_jobs.append(int(words[1][:-1]))
            add_str= '_resc'
        print good_jobs
        PROCESS_DIR =  os.path.join(os.path.dirname(os.path.realpath(rescue_file)), '..')
        print('Using old Process Dir {}'.format(PROCESS_DIR))
    else:
        add_str = ''
        good_jobs = []
        PROCESS_DIR = os.path.join(dataset_parser.get("Basics", "dagman_folder"), today)
    if not os.path.exists(PROCESS_DIR):
        os.makedirs(PROCESS_DIR)

    WORKDIR = os.path.join(PROCESS_DIR, "jobs/")
    dirname =os.path.abspath(os.path.dirname(__file__))
    if args['gpu']:
        script = os.path.join(dirname,'icetray_env_gpu.sh')
    else:
        script = os.path.join(dirname,'icetray_env.sh')
    print('Submit Script:\n {}'.format(script))
    dag_name = args["name"] + add_str
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
    RAM_str = "{} GB".format(args["request_RAM"])
    if args['gcd'] is not None:
        arguments = " --files $(PATHs) --dataset_config $(DATASET) --outfile $(OFILE) --gcd {}".format(args['gcd'])
    else:
        arguments = " --files $(PATHs) --dataset_config $(DATASET) --outfile $(OFILE) "
    print(arguments)
    submitFileContent = {"notification": "Error",
                         "log": "$(LOGFILE).log",
                         "output": "$(STREAM).out",
                         "error": "$(STREAM).err",
                         "initialdir" : "/home/tglauch/",
                         #"Requirements" : "HAS_CVMFS_icecube_opensciencegrid_org",
    #                    "Requirements" : '(Machine != "n-15.icecube.wisc.edu")',
                         #"Requirements" : '(CUDARuntimeVersion >= 10.0)',
                         "request_memory": RAM_str,
                         "request_cpus" : 1,
                         "arguments": arguments}
    if args['gpu']:
        submitFileContent["request_gpus"] = 1
        submitFileContent["Requirements"] = 'CUDACapability && has_avx'
    submitFile = pydag.htcondor.HTCondorSubmit(submitFile,
                                               script,
                                               **submitFileContent)

    submitFile.dump()
    folderlist = dataset_parser.get("Basics", "folder_list")
    basepath = [dataset_parser['Basics'][key] for key in
                dataset_parser['Basics'].keys() if 'mc_path' in key]
    filelist = dataset_parser.get("Basics", "file_list")
    
    run_filelist, num_files  = get_files_from_folder(basepath, folderlist, args['compression_format'], filelist,
                                                     args['must_contain'], args['exclude'])
    lengths = [len(i) for i in run_filelist]
    print(lengths)
    run_filelist = np.concatenate(run_filelist)
    if args['muongun'] is not None:
        with open(args['muongun'], 'w+') as f:
           f.write('\n'.join(list(run_filelist)))
        exit()
    run_filelist = [run_filelist[i:i+args['files_per_job']] for i in np.arange(0, len(run_filelist),
                    args['files_per_job'])] 
    
    nodes = []
#        if os.path.exists(outfolder):
#            shutil.rmtree(outfolder)
    if not os.path.exists(os.path.join(outfolder, 'logs')):
         os.makedirs(os.path.join(outfolder, 'logs'))
    print('Number of Jobs {}'.format(len(run_filelist)-len(good_jobs)))
    for i in range(len(run_filelist)):
        if i in good_jobs:
            continue
        fname = 'File_{}'.format(i)
        logfile = os.path.join(log_path,fname)
        stream = os.path.join(dataset_parser.get('Basics', 'out_folder'), 'logs', fname)
        PATH = ' '.join(run_filelist[i])
        outfile = os.path.join(outfolder, fname +'.npy')
        dagArgs = pydag.dagman.Macros(LOGFILE=logfile,
                                      PATHs=PATH,
                                      STREAM = stream,
                                      DATASET=args['dataset_config'],
                                      OFILE=outfile,)
        node = pydag.dagman.DAGManNode(i, submitFile)
        node.keywords["VARS"] = dagArgs
        nodes.append(node)
    dag = pydag.dagman.DAGManJob(dagFile, nodes)
    dag.dump()
    os.system("condor_submit_dag -f " + dagFile)
    time.sleep(1)
