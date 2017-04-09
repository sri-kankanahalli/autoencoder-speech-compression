# ==========================================================================
# miscellaneous utility functions
# ==========================================================================

import numpy as np
import math
import os

# MSE between two numpy arrays of the same size
def mse(a, b):
    return ((a - b) ** 2).mean(axis = None)
    
# average error betwene two numpy arrays of the same size
def avgErr(a, b):
    return (abs(a - b)).mean(axis = None)

# interleave numpy arrays of the same size along the first axis
def interleave(arr):    
    num = len(arr)
    
    r = np.empty(arr[0].shape)
    r = np.repeat(r, num, axis = 0)
    
    for i in xrange(0, num):
        r[i::num] = arr[i]
    
    return r
