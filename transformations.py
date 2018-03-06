#!/usr/bin/env python
# coding: utf-8

import numpy as np
from keras.utils import to_categorical
from scipy.stats import norm

def identity(x):
    return x

def centralize(x):
    if np.std(x)>0.:
        return ((x - np.mean(x)) / np.std(x))
    else:
        return (x - np.mean(x))

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

def oneHotEncode(x):
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
    return onehot_encoded

def oneHotEncode_noDoubleBang(x):
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

#def time_interval_0.1_to_0.9(x):
#    interval = np.percentile(x, 90)-np.percentile(x, 10)
#    return interval
