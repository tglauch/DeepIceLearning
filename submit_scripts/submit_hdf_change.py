import os
import numpy as np
import argparse
from workload_managers import *
import time


# arguments given in the terminal
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outfolder",
        help="The outfolder...",
        type=str, required=True)
    parser.add_argument(
        "--datadir",
        help="Path to a filelist to be processed",
        type=str, required=True)
    parser.add_argument(
        "--num_files",
        help="Number of files to be processed at the same time",
        type=int, default=1)
    args = parser.parse_args()
    return args
args = parseArguments().__dict__

ncpus = 1
memory = 2
condor_folder =  '/data/user/tglauch/condor/hdf_change/'
log_folder = '/scratch/tglauch/hdf_change/'

with open('submit.info', 'r') as f:
    submit_info = f.read()
file_list = sorted([os.path.join(args['datadir'],i) for i in os.listdir(args['datadir']) if i[-3:]=='.h5'])
split_list = [file_list[i:i+args['num_files']] for i in np.arange(0, len(file_list), args['num_files'])]

if not os.path.exists(condor_folder):
    os.makedirs(condor_folder)
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
if not os.path.exists(args['outfolder']):
    os.makedirs(args['outfolder'])

sargs = ' --filelist {} --outfile {}'
for i,k in enumerate(split_list):
    ofile = os.path.join(args['outfolder'], 'File_{}.h5'.format(i))
    fargs = sargs.format(' '.join(k), ofile)
    print(fargs)
    fsubmit_info = submit_info.format(ncpus=ncpus, mem=memory,
                                      args=fargs, fcondor=condor_folder,
                                      flog=log_folder,
                                      fname='File_{}'.format(i))
    submit_file = os.path.join(condor_folder, 'submit.sub')
    with open(submit_file, "wc") as sufile:
        sufile.write(fsubmit_info)
        print '\n\n\n'
    os.system("condor_submit {}".format(submit_file))
    time.sleep(0.2)
