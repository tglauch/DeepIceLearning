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
