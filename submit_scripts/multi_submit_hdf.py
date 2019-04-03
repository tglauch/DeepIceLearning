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
        help="main config file, user-specific",
        type=str, required=False)
    parser.add_argument(
        "--filelist",
        help="Path to a filelist to be processed",
        type=str, required=False)
    parser.add_argument(
        "--filename",
        help="Name of the outfile",
        type=str, required=False)
    parser.add_argument(
        "--num_files",
        help="frac of the data that should be procesed",
        type=int, required=False)
    parser.add_argument(
        "--partnumber",
        help="number of part that should be processed",
        type=int, required=False)
    args = parser.parse_args()
    return args
args = parseArguments().__dict__


request_gpus = 1
request_memory = 1
exclude = ""
condor_out_folder = "/scratch9/mkron/data/hdf/condor"
train_location = "/scratch9/mkron/data/hdf"
thisfolder = "/scratch9/mkron/software/DeepIceLearning"

print os.listdir(args["filelist"])
file_list = [i for i in os.listdir(args['filelist']) if i[-3:]=='.h5']
n_jobs = len(file_list)/args["num_files"]
#print file_list

for k in xrange(n_jobs):
	file_listy = file_list[k*args["num_files"]:(k+1)*args["num_files"]]
	arguments = ""
	arguments += ' --outfolder {} '.format(args["outfolder"])
	arguments += ' --filelist {} '.format(' '.join(file_listy))
        arguments += ' --filename {}_{}.h5'.format(args["filename"], k)
        arguments += ' --datadir {}'.format(args["filelist"]) 


	submit_info = make_slurm("kick_type.py",\
        	                     request_gpus,\
                	             float(request_memory) * 1e3,\
                        	     condor_out_folder,\
	                             train_location,\
        	                     arguments,\
                	             thisfolder,\
                        	     exclude,\
                  	             partition="kta",
				     ex_type='python',
			             log_name='{}_{}'.format(args['filename'],k))




	#print(submit_info)
	submitfile_full = os.path.join(condor_out_folder, 'submit_{}_{}.sub'.format(args["filename"], k))
	with open(submitfile_full, "wc") as file:
    		file.write(submit_info)

	os.system("sbatch {}".format(submitfile_full))
	time.sleep(1)
