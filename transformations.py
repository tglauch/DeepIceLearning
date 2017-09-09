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
