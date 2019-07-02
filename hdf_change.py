#! /usr/bin/env python
# coding: utf-8


import h5py as h5
import os
import tables
import numpy as np
import argparse
import time
from scipy.interpolate import RectBivariateSpline

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
    args = parser.parse_args()
    return args


args = parseArguments().__dict__
print args

key = 'ic_hitdoms'

through_high_E_spline = np.load('/data/user/tglauch/DeepIceLearning/pick_probs/through_n_hit_doms.npy')[()]
through_high_E_spline = RectBivariateSpline(setNewEdges(through_high_E_spline['logE_bins']),
                                            setNewEdges(through_high_E_spline['cos_zen_bins']),
                                            through_high_E_spline['H'],
                                            kx=1, ky=1, s=0)


def pick_through(ev):
    if np.log10(ev['ic_hitdoms']) < 1.6:
        return True
    val = through_high_E_spline(np.log10(ev[key]), np.cos(ev['mc_prim_zen']))
    ran_num = np.random.uniform(0,1,1)
    if ran_num > val:
        return False
    else:
        ran_num = np.random.uniform(0,max_rand,1)
        if ran_num > picker['through']:
            return False
        else:
            return True
    
start_high_E_spline = np.load('/data/user/tglauch/DeepIceLearning/pick_probs/starting_n_hit_doms.npy')[()]
start_high_E_spline = RectBivariateSpline(setNewEdges(start_high_E_spline['logE_bins']),
                                            setNewEdges(start_high_E_spline['cos_zen_bins']),
                                            start_high_E_spline['H'],
                                            kx=1, ky=1, s=0)

start_low_E_spline = np.load('/data/user/tglauch/DeepIceLearning/pick_probs/starting_n_hit_domsfew_hits.npy')[()]
start_low_E_spline = RectBivariateSpline(setNewEdges(start_low_E_spline['logE_bins']),
                                            setNewEdges(start_low_E_spline['cos_zen_bins']),
                                            start_low_E_spline['H'],
                                            kx=1, ky=1, s=0)
def pick_start(ev):
    if np.log10(ev['ic_hitdoms']) < 1.6:
        val = start_low_E_spline(np.log10(ev[key]), np.cos(ev['mc_prim_zen']))
    else:
        val = start_high_E_spline(np.log10(ev[key]), np.cos(ev['mc_prim_zen']))
    ran_num = np.random.uniform(0,1,1)
    if ran_num > val:
        return False
    else:
        ran_num = np.random.uniform(0,max_rand,1)
        if ran_num > picker['starting']:
            return False
        else:
            return True
    
cascade_high_E_spline = np.load('/data/user/tglauch/DeepIceLearning/pick_probs/cascade_n_hit_doms.npy')[()]
cascade_high_E_spline = RectBivariateSpline(setNewEdges(cascade_high_E_spline['logE_bins']),
                                            setNewEdges(cascade_high_E_spline['cos_zen_bins']),
                                            cascade_high_E_spline['H'],
                                            kx=1, ky=1, s=0)

cascade_low_E_spline = np.load('/data/user/tglauch/DeepIceLearning/pick_probs/cascade_n_hit_domsfew_hits.npy')[()]
cascade_low_E_spline = RectBivariateSpline(setNewEdges(cascade_low_E_spline['logE_bins']),
                                            setNewEdges(cascade_low_E_spline['cos_zen_bins']),
                                            cascade_low_E_spline['H'],
                                            kx=1, ky=1, s=0)
def pick_cascade(ev):
    if np.log10(ev['ic_hitdoms']) < 1.6:
        val = cascade_low_E_spline(np.log10(ev[key]), np.cos(ev['mc_prim_zen']))
    else:
        val = cascade_high_E_spline(np.log10(ev[key]), np.cos(ev['mc_prim_zen']))
    ran_num = np.random.uniform(0,1,1)
    if ran_num > val:
        return False
    else:
        ran_num = np.random.uniform(0,max_rand,1)
        if ran_num > picker['cascade']:
            return False
        else:
            return True
        
        
passing_high_E_spline = np.load('/data/user/tglauch/DeepIceLearning/pick_probs/passing_n_hit_doms.npy')[()]
passing_high_E_spline = RectBivariateSpline(setNewEdges(passing_high_E_spline['logE_bins']),
                                            setNewEdges(passing_high_E_spline['cos_zen_bins']),
                                            passing_high_E_spline['H'],
                                            kx=1, ky=1, s=0)

passing_low_E_spline = np.load('/data/user/tglauch/DeepIceLearning/pick_probs/passing_n_hit_domsfew_hits.npy')[()]
passing_low_E_spline = RectBivariateSpline(setNewEdges(passing_low_E_spline['logE_bins']),
                                            setNewEdges(passing_low_E_spline['cos_zen_bins']),
                                            passing_low_E_spline['H'],
                                            kx=1, ky=1, s=0)

def pick_passing(ev):
    if np.log10(ev['ic_hitdoms']) < 1.6:
        val = passing_low_E_spline(np.log10(ev[key]), np.cos(ev['mc_prim_zen']))
    else:
        val = passing_high_E_spline(np.log10(ev[key]), np.cos(ev['mc_prim_zen']))
    ran_num = np.random.uniform(0,1,1)
    if ran_num > val:
        return False
    else:
        ran_num = np.random.uniform(0,max_rand,1)
        if ran_num > picker['passing']:
            return False
        else:
            return True
        
stopping_low_E_spline = np.load('/data/user/tglauch/DeepIceLearning/pick_probs/stopping_n_hit_domsfew_hits.npy')[()]
stopping_low_E_spline = RectBivariateSpline(setNewEdges(stopping_low_E_spline['logE_bins']),
                                            setNewEdges(stopping_low_E_spline['cos_zen_bins']),
                                            stopping_low_E_spline['H'],
                                            kx=1, ky=1, s=0)

def pick_stopping(ev):
    if np.log10(ev['ic_hitdoms']) > 1.6:
        return True
    else:
        val = stopping_low_E_spline(np.log10(ev[key]), np.cos(ev['mc_prim_zen']))
    ran_num = np.random.uniform(0,1,1)
    if ran_num > val:
        return False
    else:
        ran_num = np.random.uniform(0,max_rand,1)
        if ran_num > picker['passing']:
            return False
        else:
            return True


def pick_events(ev):
    if ev['classification'] not in [0,1,2,3,4,11,22,23]:
        return False
    if ev['classification'] in [1]:
        return pick_cascade(ev)
    elif ev['classification'] in [2,22]:
        return pick_through(ev)
    elif ev['classification'] in [3]:
        return pick_start(ev)
    elif ev['classification'] in [4,23]:
        return pick_stopping(ev)
    elif ev['classification'] in [0,11]:
        return pick_passing(ev)
    else:
        return False


picker = {'through': 9, 'cascade':9, 'passing':9, 'starting':9, 'stopping':10}
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
        one_h5file = h5.File(os.path.join(DATA_DIR, fili), 'r')
        num_events = len(one_h5file["reco_vals"])
        for k in np.random.choice(num_events, num_events, replace=False):
            print k
            if pick_events(one_h5file["reco_vals"][k]) == False:
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

