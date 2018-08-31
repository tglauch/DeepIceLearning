#!/usr/bin/env python
# coding: utf-8

import numpy as np
from keras.utils import to_categorical
from scipy.stats import norm

def identity(x,r_vals=None):
    return x

def centralize(x):
    if np.std(x)>0.:
        return ((x - np.mean(x)) / np.std(x))
    else:
        return (x - np.mean(x))
def waveform_offset(x):
    return (x-10000)/400

def max(x):
    return np.amax(x)

def max_min_delta(x):
    return np.max(x)-np.min(x)

def shift_min_to_zero(x):
    return x-np.amin(x)

def sort_input(x):
    return np.sort(np.ndarray.flatten(x))

def plus_one_log10(x):
    tmp = x + 1.
    return np.log10(tmp)

def zenith_prep(x, r_vals=None):
    #if r_vals is not None:
    #   print x, r_vals['zenith']
    x =  x / np.pi
    return x


def log_handle_zeros_flatten_top30(x):
    #tmp = np.where(x != 0, np.log10(x), 0)
    return np.sort(np.ndarray.flatten(np.log10(1.+x)))[-30:]

def log_handle_zeros(x):
    return np.where(x != 0, np.log10(x), 0)

def sort_input_and_top20(x):
    return np.sort(np.ndarray.flatten(x))[-20:]

def smeared_one_hot_encode_logbinned(E):
    width = 0.16
    bins=np.linspace(3,7,50)
    gauss = norm(loc = np.log10(E), scale = width)
    smeared_hot_output = gauss.pdf(bins)
    return smeared_hot_output/np.sum(smeared_hot_output)

def one_hot_encode_logbinned(x):
    bins=np.linspace(3,7,50)
    bin_indcs = np.digitize(np.log10(x), bins)
    one_hot_output = to_categorical(bin_indcs, len(bins))
    return one_hot_output

def zenith_to_binary(x):
    """
    returns boolean values for the zenith (0 or 1; up or down, > or < pi/2) as np.array.
    """
    ret = np.copy(x)
    ret[ret < 1.5707963268] = 0.0
    ret[ret > 1] = 1.0
    return ret

def time_prepare(x):
    """
    This function normalizes the finite values of input data to the interval [0,1] and 
    replaces all infinity-values with replace_with (defaults to 1).
    """
    replace_with = 1.0
    ret = np.copy(x)
    time_np_arr_max = np.max(ret[ret != np.inf])
    time_np_arr_min = np.min(ret)
    ret = (ret - time_np_arr_min) / (time_np_arr_max - time_np_arr_min)
    ret[ret == np.inf] = replace_with
    return ret

def oneHotEncode_EventType_simple(x):
    """
    This function one hot encodes the input for the event types cascade, tracks, doubel-bang
    """
    # define universe of possible input values
    onehot_encoded = []
    # universe has to defined depending on the problem, in this implementation integers are neccesary
    universe = [1, 2, 3]
    for i in range(len(universe)):
        if x == universe[i]:
            value = 1.
        else:
            value = 0.
        onehot_encoded.append(value)
    return onehot_encoded

def oneHotEncode_EventType_noDoubleBang_simple(x):
    """
    This function one hot encodes the input
    """
    # define universe of possible input values
    onehot_encoded = []
    # universe has to defined depending on the problem, in this implementation integers are neccesary
    universe = [1, 2, 3]
    for i in range(len(universe)):
        if x == universe[i]:
            value = 1.
        else:
            value = 0.
        onehot_encoded.append(value)
    if onehot_encoded == [0., 0., 1.]:
        onehot_encoded = [1.0, 0.0, 0.0]
    return onehot_encoded[:-1]

def log_of_sum(x):
    return np.log10(np.sum(x)+0.0001)

def max_min_delta_log(x):
    return np.log10(np.max(x)-np.min(x))


def oneHotEncode_01(x, r_vals=None):
    """
    This function one hot encodes the input for a binary label 
    """
    # define universe of possible input values
    onehot_encoded = []
    # universe has to defined depending on the problem, in this implementation integers are neccesary
    universe = [0, 1]
    for i in range(len(universe)):
        if x == universe[i]:
            value = 1.
        else:
            value = 0.
        onehot_encoded.append(value)
    return onehot_encoded


def oneHotEncode_EventType_exact(x, r_vals=None):
    """
    This function one hot encodes the input for the event types cascade, tracks, doubel-bang
    """
    # define universe of possible input values
    onehot_encoded = []
    # universe has to defined depending on the problem, in this implementation integers are neccesary
    universe = [0, 1, 2, 3, 4, 5, 6]
    for i in range(len(universe)):
        if x == universe[i]:
            value = 1.
        else:
            value = 0.
        onehot_encoded.append(value)
    return onehot_encoded

def oneHotEncode_EventType(x, r_vals=None):
    """
    This function one hot encodes the input for the event types cascade, tracks, doubel-bang
    """
    # define universe of possible input values
    #fail = [0., 0., 0.]
    cascade = [1., 0., 0.]
    track = [0., 1., 0.]
    s_track = [0., 0., 1.]
    # map x to possible classes
    mapping = {0:cascade, 1:cascade, 2:track, 3:track, 4:track, 5:doublebang, 6:doublebang, 7:cascade, 8:track, 9:cascade}
    return mapping[x]

def oneHotEncode_EventType_stratingTrack(x, r_vals=None):
    """
    This function one hot encodes the input for the event types cascade, tracks, doubel-bang, starting tracks
    """
    #print type(list(r_vals))
    #print "r_vals: {}".format(r_vals)
    #print "x: {}".format(x)
    # define universe of possible input values
    #fail = [0., 0., 0., 0.]
    cascade = [1., 0., 0., 0.]
    track = [0., 1., 0., 0.]
    doublebang = [0., 0., 1., 0.]
    startingTrack = [0., 0., 0., 1.]
    # map x to possible classes
    mapping = {0:cascade, 1:cascade, 2:track, 3:startingTrack, 4:track, 5:doublebang, 6:doublebang, 7:cascade, 8:track, 9:cascade}
    return mapping[x]

def oneHotEncode_Starting_padding(x, r_vals):
    pos = [r_vals[14], r_vals[15], r_vals[16]]
    dir = [r_vals[17], r_vals[18], r_vals[19]]
    gcdfile = "/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2013.56429_V0.i3.gz"
    padding = 100
    surface = icecube.MuonGun.ExtrudedPolygon.from_file(gcdfile, padding= padding)
    intersections = surface.intersection(pos, dir)
    if intersections.first <= 0 and intersections.second > 0:
        starting = 0  # starting event
    else:
        starting = 1  # through-going or stopping event
    return starting

def oneHotEncode_DB(x, r_vals=None):
    """
    This function one hot encode for event type  doubel-bang, no-double bang
    """
    #print type(list(r_vals))
    #print "r_vals: {}".format(r_vals)
    #print "x: {}".format(x)
    # define universe of possible input values
    fail = [0., 0.]
    ndoublebang = [1., 0.]
    doublebang = [0., 1.,]
    
    # map x to possible classes
    if x == 5: #Double Bang
        onehot_encoded = doublebang
    elif x in [0,1,2,3,4,6,7,8,9]:
        onehot_encoded = ndoublebang
    else:
        onehot_encoded = fail
    return onehot_encoded
