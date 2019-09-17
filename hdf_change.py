#! /usr/bin/env python
# coding: utf-8


import h5py as h5
import os
import tables
import numpy as np
import argparse
import time

def setNewEdges(edges):
    newEdges = []
    for i in range(0,len(edges)-1):
        newVal = (edges[i]+edges[i+1])*1.0/2
        newEdges.append(newVal)
    return np.array(newEdges)


# arguments given in the terminal
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outfile",
        help="main config file, user-specific",
        type=str)
    parser.add_argument(
        "--filelist",
        help="Path to a filelist to be processed",
        type=str, nargs="+")
    parser.add_argument(
        "--datadir",
        help=" data directory",
        type=str)
    parser.add_argument(
        "--picker",
        help="which picker function to use",
        type=str, default='mu_e_reco')
    args = parser.parse_args()
    return args


args = parseArguments().__dict__
print args
exec 'import pickers.{} as picker'.format(args['picker'])
input_shape = [10, 10, 60]
FILTERS = tables.Filters(complib='zlib', complevel=9)

if args['filelist'] is not None:
    file_list = args["filelist"]
    print file_list
elif args["datadir"] is not None:
    DATA_DIR = args["datadir"]
    file_list = [i for i in os.listdir(DATA_DIR) if '.h5' in i]
tfile = file_list[0]
print('Try to open {}'.format(tfile))
hf1 = h5.File(tfile, 'r')
keys = hf1.keys()
dtype=hf1["reco_vals"].dtype

with tables.open_file(args['outfile'], mode="w", title="Events for training the NN",
                      filters=FILTERS) as h5file:
    input_features = []
    for okey in keys[:-1]:
        feature = h5file.create_earray(
                h5file.root, okey, tables.Float64Atom(),
                (0, input_shape[0], input_shape[1], input_shape[2], 1),
                title=okey)
        feature.flush()
        input_features.append(feature)
    reco_vals = tables.Table(h5file.root, 'reco_vals', description=dtype)
    h5file.root._v_attrs.shape = input_shape
    hf1.close()
 
    a= time.time()
    for fili in file_list:
        print('Open {}'.format(fili))
        one_h5file = h5.File(fili, 'r')
        num_events = len(one_h5file["reco_vals"])
        for k in np.random.choice(num_events, num_events, replace=False):
            print k
            if picker.pick_events(one_h5file["reco_vals"][k]) == False:
                print('continue')
                continue             
            for i, okey in enumerate(keys[:-1]):
                input_features[i].append(np.expand_dims(one_h5file[okey][k], axis=0))
            reco_vals.append(np.atleast_1d(one_h5file["reco_vals"][k]))
        for inp_feature in input_features:
                inp_feature.flush()
        reco_vals.flush()
    print('Time for one File {}'.format(time.time()-a))
    one_h5file.close()
h5file.close()

