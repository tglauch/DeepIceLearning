#!/usr/bin/env python
# coding: utf-8

import numpy as np
from keras.utils import to_categorical

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

def sort_input(x):
    return np.sort(np.ndarray.flatten(x))

def sort_input_and_top20(x):
    return np.sort(np.ndarray.flatten(x))[-20:]

def one_hot_encode_logbinned(x):
    bins=np.linspace(3,7,40)
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
