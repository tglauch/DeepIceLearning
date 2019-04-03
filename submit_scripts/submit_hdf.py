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
        type=str, nargs="+", required=False)
    parser.add_argument(
        "--filename",
        help="Name of the outfile",
        type=str, required=False)
    parser.add_argument(
        "--scale",
        help="frac of the data that should be procesed",
        type=int, required=False)
    parser.add_argument(
        "--partnumber",
        help="number of part that should be processed",
        type=int, required=False)
    args = parser.parse_args()
    return args
args = parseArguments().__dict__


exclude = "bigbird"
request_gpus = 1
request_memory = 4

condor_out_folder = "/scratch9/mkron/data/hdf/condor"
train_location = "/scratch9/mkron/data/hdf"
thisfolder = "/scratch9/mkron/software/DeepIceLearning"

arguments = ""
arguments += ' --outfolder {} '.format(args["outfolder"])
arguments += ' --filename {} '.format(args["filename"])

exclude = ""


file_list = []
for (dirpath, dirnames, filenames) in os.walk(args["filelist"][0]):
    file_list.extend(filenames)
    break
steps= len(file_list)/args["scale"]
print file_list
file_listy = file_list[(args["partnumber"]-1)*steps:args["partnumber"]*steps]
print file_listy
arguments += ' --filelist {} '.format(' '.join(file_listy))



submit_info = make_slurm("shrink_hdf.py",\
                             request_gpus,\
                             float(request_memory) * 1e3,\
                             condor_out_folder,\
                             train_location,\
                             arguments,\
                             thisfolder,\
                             exclude,\
                             partition="kta",
			     ex_type='python')




print(submit_info)
submitfile_full = os.path.join(condor_out_folder, 'submit.sub')
with open(submitfile_full, "wc") as file:
    file.write(submit_info)

os.system("sbatch {}".format(submitfile_full))
time.sleep(3)
