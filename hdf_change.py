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
    args = parser.parse_args()
    return args

def sta_prob(ev):
    dep_e = np.log10(ev['e_dep'])
    return np.min([(dep_e)**3/27., 1])

def scs_prob(ev):
    dep_e = np.log10(ev['e_dep'])
    return np.min([(dep_e)**2/9., 1])

args = parseArguments().__dict__
print args

picker = {0:6, 1:6, 2:5, 3:12, 4: 12}
reweight = {0:None, 1:scs_prob, 2:None, 3:sta_prob, 4:None}
retag = {11: 0}

max_rand = np.max([picker[i] for i in picker.keys()])
print('max_rand {}'.format(max_rand))
DATA_DIR = args["datadir"]
input_shape = [10, 10, 60]
FILTERS = tables.Filters(complib='zlib', complevel=9)

if args['filelist'] is not None:
    file_list = args["filelist"]
    print file_list
elif DATA_DIR is not None:
    file_list = [i for i in os.listdir(DATA_DIR) if '.h5' in i]
tfile = os.path.join(DATA_DIR, file_list[0])
print('Try to open {}'.format(tfile))
hf1 = h5.File(tfile, 'r')
keys = hf1.keys()
dtype=hf1["reco_vals"].dtype

with tables.open_file(args['outfile'], mode="w", title="Events for training the NN",
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
    hf1.close()
 
    a= time.time()
    for fili in file_list:
        print('Open {}'.format(fili))
        one_h5file = h5.File(os.path.join(DATA_DIR, fili), 'r')
        for k in xrange(len(one_h5file["reco_vals"])):
            print k
            classi = one_h5file["reco_vals"][k]['classification']
            if classi in retag.keys():
                classi = retag[classi]
            if classi not in picker.keys():
                print('continue')
                continue
            if classi in [3, 4] and one_h5file["reco_vals"][k]['track_length'] < 100:
                continue
            if reweight[classi] is not None:
                prob_val = reweight[classi](one_h5file["reco_vals"][k])
                if prob_val < float(np.random.uniform(0,1,1)):
                    continue
            rand = np.random.choice(np.arange(0,max_rand))
            if picker[classi] < rand:
                continue
            for i, key in enumerate(keys[:-1]):
                input_features[i].append(np.expand_dims(one_h5file[key][k], axis=0))
            reco_vals.append(np.atleast_1d(one_h5file["reco_vals"][k]))
        for inp_feature in input_features:
                inp_feature.flush()
        reco_vals.flush()
    print('Time for one File {}'.format(time.time()-a))
    one_h5file.close()
h5file.close()

