# ==========================================================================
# neural network Keras utility functions
# ==========================================================================
import numpy as np
import h5py
import os

# hackily patches a really stupid TensorFlow bug affecting K.reshape
import tensorflow as tf
from keras import backend as K

def new_reshape(x, shape):
    fixed_shape = tuple([int(w) for w in shape])
    return tf.reshape(x, fixed_shape)

K.reshape = new_reshape
from keras.models import Model

# given an [array] of scalar quantization bins, finds the
# closest one to [value]
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]







