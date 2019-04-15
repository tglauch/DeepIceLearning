import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import h5py
import os

def setNewEdges(edges):
    newEdges = []
    for i in range(0,len(edges)-1):
        newVal = (edges[i]+edges[i+1])*1.0/2
        newEdges.append(newVal)
    return np.array(newEdges)

def powerlaw(mc_path, mc_loc,  gamma=1.0, keys={'ow': 'generator_ow', 'trueE' : 'true_e'}):
    norm = []
    for i in mc_path:
        x= h5py.File(os.path.join(mc_loc,i), 'r')
        norm.append(x['reco_vals'][keys['ow']] * (x['reco_vals'][keys['trueE']]) ** (-gamma))
        x.close()
    N = len(np.concatenate(norm))
    norm = 1. * np.sum(np.concatenate(norm))
    func = lambda mc : 1. * N / norm * mc[keys['ow']] * (mc[keys['trueE']]) ** (-gamma)
    return func


def weight_from_mc_fit(mc_path, mc_loc,  bins=100, key='trunc_e'):
    reco_val = []
    for i in mc_path:
        x= h5py.File(os.path.join(mc_loc,i), 'r')
        reco_val.append(x['reco_vals'][key])
        x.close()
    reco_val = np.concatenate(reco_val)
    vals, bins = np.histogram(reco_val, bins=100)
    interp = InterpolatedUnivariateSpline(setNewEdges(bins), vals, k=1)
    func = lambda mc : 1./interp(mc[key])
    return func


def scale_w_logE_pow3(mc_path, mc_loc, key='trunc_e'):
    norm = []
    for i in mc_path:
        x= h5py.File(os.path.join(mc_loc,i), 'r')
        vals = x['reco_vals'][key]
        norm.append(np.log10(np.max([vals, np.ones(len(vals)) * 100], axis=0))**3)
        x.close()
    N = len(np.concatenate(norm))
    norm = 1. * np.sum(np.concatenate(norm))
    func = lambda mc : 1. * N / norm * (np.log10(np.max([mc[key], np.ones(len(mc[key])) * 100], axis=0)))**3
    return func
