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

# hackily patches a really stupid TensorFlow bug affecting K.reshape
import tensorflow as tf
from keras import backend as K

def new_reshape(x, shape):
    fixed_shape = tuple([int(w) for w in shape])
    return tf.reshape(x, fixed_shape)

K.reshape = new_reshape










