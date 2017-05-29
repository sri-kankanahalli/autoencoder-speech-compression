# ==========================================================================
# neural network Keras blocks / Theano operations needed for models, as well
# as a few utility functions
# ==========================================================================

import numpy as np
from keras.models import *
from keras.layers import *
from keras.layers.core import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras import backend as K
from keras.regularizers import *
import theano.tensor as T
import theano

from consts import *

# ---------------------------------------------------
# "Phase shift" upsampling layer, as discussed in [that one
# superresolution paper]
#
# Takes vector of size: B x S  x nC
# And returns vector:   B x nS x C
# ---------------------------------------------------
class PhaseShiftUp1D(Layer):
    def __init__(self, n, **kwargs):
        super(PhaseShiftUp1D, self).__init__(**kwargs)
        self.n = n
    
    def build(self, input_shape):
        # no trainable parameters
        self.trainable_weights = []
        super(PhaseShiftUp1D, self).build(input_shape)
        
    def call(self, x, mask=None):
        r = T.reshape(x, (x.shape[0], x.shape[1], x.shape[2] / self.n, self.n))
        r = T.transpose(r, (0, 1, 3, 2))
        r = T.reshape(r, (x.shape[0], x.shape[1] * self.n, x.shape[2] / self.n))
        return r
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * self.n, input_shape[2] / self.n)

    def get_config(self):
        config = {
            'n' : self.n
        }
        base_config = super(PhaseShiftUp1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



# ---------------------------------------------------
# Different types of "blocks" that make up all of our
# models
# ---------------------------------------------------

# activation used in all blocks
def activation():
    return LeakyReLU(0.3)

# residual block, going from NCHAN to NCHAN channels
def residual_block(num_chans, filt_size, dilation = 1):
    def f(input):
        shortcut = input
        
        res = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'tanh',
                     dilation_rate = dilation)(input)
        
        gate = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'sigmoid',
                     dilation_rate = dilation)(input)
        
        sig = Multiply()([shortcut, gate])
        m = Add()([sig, res])
        return m
    
    return f


# change number of channels to to_chan via convolution
def channel_change_block(to_chan, filt_size, last_activ = True):
    def f(input):
        shortcut = Permute((2, 1))(input)
        shortcut = GlobalAveragePooling1D()(shortcut)
        shortcut = Reshape((1, -1))(shortcut)
        shortcut = UpSampling1D(to_chan)(shortcut)
        shortcut = Permute((2, 1))(shortcut)
 
        res = Conv1D(to_chan, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'tanh')(input)

        gate = Conv1D(to_chan, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'sigmoid')(input)

        sig = Multiply()([shortcut, gate])
        m = Add()([sig, res])
        return m
        
    return f


# downsample the signal 2x
def downsample_block(num_chans, filt_size):
    def f(input):
        shortcut = AveragePooling1D(2)(input)
        
        res = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'tanh',
                     strides = 2)(input)

        gate = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'sigmoid',
                     strides = 2)(input)

        sig = Multiply()([shortcut, gate])
        m = Add()([sig, res])
        return m
    
    return f


# upsample the signal 2x
def upsample_block(num_chans, filt_size):
    def f(input):
        shortcut = UpSampling1D(2)(input)
        
        res = Conv1D(num_chans * 2, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'tanh')(input)
        res = PhaseShiftUp1D(2)(res)

        gate = Conv1D(num_chans * 2, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'sigmoid')(input)
        gate = PhaseShiftUp1D(2)(gate)

        sig = Multiply()([shortcut, gate])
        m = Add()([sig, res])
        return m
    
    return f


# ---------------------------------------------------
# Utility / loss functions
# ---------------------------------------------------

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

# euclidean distance "layer"
def EuclideanDistance():
    def func(vects):
        x, y = vects
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True) + K.epsilon())

    def shape(shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    return Lambda(func, output_shape = shape)

# correlation loss function for samples of one variable
def correlation(var1, var2):
    var1 = K.flatten(var1)
    var2 = K.flatten(var2)
    
    covariance = K.mean(var1 * var2) - K.mean(var1) * K.mean(var2)
    denom = K.std(var1) * K.std(var2) + K.epsilon()
    correlation = covariance / denom
    return correlation




