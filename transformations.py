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
    y = np.ndarray.flatten(x)
    z = np.sort(y)
    return z
