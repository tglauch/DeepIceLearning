#!/usr/bin/env python
# coding: utf-8

import numpy as np


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
