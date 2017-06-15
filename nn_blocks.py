# ==========================================================================
# neural network Keras blocks / Theano operations needed for models, as well
# as a few utility functions
# ==========================================================================

import numpy as np

from consts import *
from utility import *
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.layers.core import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.regularizers import *
from keras.initializers import *

# ---------------------------------------------------
# 1D "phase shift" upsampling layer, as discussed in [that one
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
        r = K.reshape(x, (-1, x.shape[1], x.shape[2] // self.n, self.n))
        r = K.permute_dimensions(r, (0, 1, 3, 2))
        r = K.reshape(r, (-1, x.shape[1] * self.n, x.shape[2] // self.n))
        return r
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * self.n, input_shape[2] // self.n)

    def get_config(self):
        config = {
            'n' : self.n,
        }
        base_config = super(PhaseShiftUp1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# ---------------------------------------------------
# 1D linear upsampling layer (upsamples by linear interpolation)
#
# Takes vector of size: B x S  x C
# And returns vector:   B x 2S x C
# ---------------------------------------------------
class LinearUpSampling1D(Layer):
    def __init__(self, fmt = None, **kwargs):
        super(LinearUpSampling1D, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # no trainable parameters
        self.trainable_weights = []
        super(LinearUpSampling1D, self).build(input_shape)
        
    def call(self, x, mask=None):
        u = K.repeat_elements(x, 2, axis = 1)
        u = (u[:, :-1] + u[:, 1:]) / 2.0
        u = K.concatenate((u, u[:, -1:]), axis = 1)

        return u
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * 2, input_shape[2])

    def get_config(self):
        config = {}
        base_config = super(LinearUpSampling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# ---------------------------------------------------
# 1D "channel resize" layer
#
# Takes vector of size: B x S x oldC
# And returns vector:   B x S x newC
# ---------------------------------------------------
class ChannelResize1D(Layer):
    def __init__(self, nchans, **kwargs):
        super(ChannelResize1D, self).__init__(**kwargs)
        self.nchans = nchans
    
    def build(self, input_shape):
        # no trainable parameters
        self.trainable_weights = []
        super(ChannelResize1D, self).build(input_shape)
        
    def call(self, x, mask=None):
        c = K.mean(x, axis = 2)
        c = K.expand_dims(c, axis = 2)
        c = K.repeat_elements(c, self.nchans, axis = 2)
        return c
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.nchans)

    def get_config(self):
        config = {
            'nchans' : self.nchans,
        }
        base_config = super(ChannelResize1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# ---------------------------------------------------
# Residual "block" that makes up all of our models
# ---------------------------------------------------

# activation used in all blocks
def activation(init = 0.3):
    # input is of form [nbatch x channel_size x num_channels],
    # so we share axis 1
    return PReLU(alpha_initializer = Constant(init),
                 shared_axes = [1])

# super advanced residual block, supporting the following additional
# operations
#     - upsampling
#     - downsampling
#     - channel_change
# as well as a gating operation and dilated convolutions
def residual_block(num_chans, filt_size, dilation = 1, gate = False,
                   operation = 'none'):
    def f(inp):
        # ---------------------------------------
        # shortcut connection
        # ---------------------------------------
        shortcut = inp
        if (operation == 'upsample'):
            shortcut = LinearUpSampling1D()(shortcut)
        elif (operation == 'downsample'):
            shortcut = AveragePooling1D()(shortcut)
        elif (operation == 'channel_change'):
            shortcut = ChannelResize1D(num_chans)(shortcut)

        # ---------------------------------------
        # residual operation
        # ---------------------------------------
        res = inp

        # conv1
        if (operation == 'upsample'):
            conv1_nc = num_chans * 2
            conv1_stride = 1
        elif (operation == 'downsample'):
            conv1_nc = num_chans
            conv1_stride = 2
        else:
            conv1_nc = num_chans
            conv1_stride = 1

        res = Conv1D(conv1_nc, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear',
                     dilation_rate = dilation,
                     strides = conv1_stride)(res)
        res = activation(0.3)(res)

        # conv2
        res = Conv1D(conv1_nc, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear',
                     dilation_rate = dilation)(res)
        if (operation == 'upsample'):
            res = PhaseShiftUp1D(2)(res)
        res = activation(0.3)(res)

        if (operation != 'none'):
            return res

        # ---------------------------------------
        # gating (if enabled)
        # ---------------------------------------
        if (gate):
            shortcut_gate = Conv1D(conv1_nc, 3, padding = 'same',
                                   kernel_initializer = W_INIT,
                                   bias_initializer = Constant(3),
                                   activation = 'sigmoid',
                                   dilation_rate = dilation,
                                   strides = conv1_stride)(inp)
            if (operation == 'upsample'):
                shortcut_gate = PhaseShiftUp1D(2)(shortcut_gate)
            shortcut = Multiply()([shortcut, shortcut_gate])

            res_gate = Lambda(lambda x : 1.0 - x,
                              output_shape = lambda s : s)(shortcut_gate)
            res = Multiply()([res, res_gate])

        # ---------------------------------------
        # final output
        # ---------------------------------------
        return Add()([shortcut, res])
    
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




