#! /usr/bin/env python
# coding: utf-8


import h5py as h5
import os
import tables
import numpy as np
import argparse
import time


# arguments given in the terminal
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outfolder",
        help="main config file, user-specific",
        type=str)
    parser.add_argument(
        "--filelist",
        help="Path to a filelist to be processed",
        type=str, nargs="+")
    parser.add_argument(
        "--filename",
        help="Name of the outfile",
        type=str)
    parser.add_argument(
        "--datadir",
        help=" data directory",
        type=str)

    args = parser.parse_args()
    return args
args = parseArguments().__dict__


SAVE_DIR = args['outfolder']
outfile_name = args["filename"]
DATA_DIR = args["datadir"]
input_shape = [10, 10, 60]
FILTERS = tables.Filters(complib='zlib', complevel=9)
picker_dict = {0:20, 1:75, 2:52, 3:30, 4:50, 5:100, 6:100, 7:100, 8:100, 9:100}
db_picker = {0:1, 1:100} # 0 = kuerzer als 5, 10 meter


file_list = args["filelist"]
print file_list
print type(file_list)

outfile = os.path.join(SAVE_DIR, outfile_name)
hf1 = h5.File(os.path.join(DATA_DIR, file_list[0]))
keys = hf1.keys()
dtype=hf1["reco_vals"].dtype

with tables.open_file(outfile, mode="w", title="Events for training the NN",
                      filters=FILTERS) as h5file:
    input_features = []
    for key in keys[:-1]:
        feature = h5file.create_earray(
                h5file.root, key, tables.Float64Atom(),
                (0, input_shape[0], input_shape[1], input_shape[2], 1),
                title=key)
        feature.flush()
        input_features.append(feature)
    reco_vals = tables.Table(h5file.root, 'reco_vals', description=dtype)
    h5file.root._v_attrs.shape = input_shape
    
    a= time.time()
    for fili in file_list:
        one_h5file = h5.File(os.path.join(DATA_DIR, fili))
        for k in xrange(len(one_h5file["reco_vals"])):
            class_n = one_h5file["reco_vals"][k]["ClassificationLabel"]
            if class_n in [5, 6]:
                tau_decay_length = one_h5file["reco_vals"][k]["TauDecayLength"]
                if tau_decay_length >= 5:
                    cut = db_picker[1]
                else:
                    cut = db_picker[0]
            elif class_n in [0, 1]:
                primary_energy = one_h5file["reco_vals"][k]["energyFirstParticle"]
                max = 7.69
                min = 3.69
                m = (99/(max-min))
                b = 60 - (m * min)
                cut = m * np.log10(primary_energy) + b # percentage that is taken
            else:
                cut = picker_dict[class_n]
            rand = np.random.choice(range(1, 101))
            if rand <= cut:
                for i, key in enumerate(keys[:-1]):
                    input_features[i].append(np.expand_dims(one_h5file[key][k], axis=0))
                reco_vals.append(np.atleast_1d(one_h5file["reco_vals"][k]))
    for inp_feature in input_features:
            inp_feature.flush()
    reco_vals.flush()
h5file.close()

print time.time()-a
