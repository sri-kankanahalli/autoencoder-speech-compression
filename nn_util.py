# ==========================================================================
# neural network Keras utility / loss functions
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

# quantization utility functions
def unquantize_batch(one_hot):
    from consts import QUANT_BINS
    out = K.dot(K.variable(one_hot), K.expand_dims(QUANT_BINS))
    out = K.reshape(out, (out.shape[0], -1))
    return K.eval(out)

def unquantize_vec(one_hot):
    from consts import QUANT_BINS
    out = K.dot(K.variable(one_hot), K.expand_dims(QUANT_BINS))
    out = K.reshape(out, (-1,))
    return K.eval(out)

# NaN-safe RMSE loss function
def rmse(y_true, y_pred):
    mse = K.mean(K.square(y_pred - y_true), axis=-1)
    return K.sqrt(mse + K.epsilon())

# function to freeze weights
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
        if (l is Model):
            make_trainable(l, val)

# given an [array] of scalar quantization bins, finds the
# closest one to [value]
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]



