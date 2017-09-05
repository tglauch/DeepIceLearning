#!/usr/bin/env python
# coding: utf-8

import numpy as np


def identity(x):
    return x


def centralize(x):
    return ((x - np.mean(x)) / np.std(x))
