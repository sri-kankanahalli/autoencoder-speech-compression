# ==========================================================================
# neural network Keras layers / blocks needed for models, as well
# as a few utility functions
# ==========================================================================

import numpy as np

from consts import *
from nn_util import *
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.layers.core import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.regularizers import *
from keras.initializers import *
from keras.activations import softmax

# weight initialization used in all layers of network
W_INIT = Orthogonal()

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
# Scalar quantization / dequantization layers
# ---------------------------------------------------

# both layers rely on the shared [QUANT_BINS] variable in consts.py

# quantization: takes in    [BATCH x WINDOW_SIZE]
#               and returns [BATCH x WINDOW_SIZE x NBINS]
# where the last dimension is a one-hot vector of bins
#
# [bins initialization is in consts.py]
class SoftmaxQuantization(Layer):
    def __init__(self, **kwargs):
        super(SoftmaxQuantization, self).__init__(**kwargs)
   
    def build(self, input_shape):
        self.SOFTMAX_TEMP = K.variable(500.0)
        self.trainable_weights = [QUANT_BINS,
                                  self.SOFTMAX_TEMP]
        super(SoftmaxQuantization, self).build(input_shape)
        
    def call(self, x, mask=None):
        # x is an array: [BATCH x WINDOW_SIZE]
        # x_r becomes:   [BATCH x WINDOW_SIZE x 1]
        x_r = K.reshape(x, (-1, x.shape[1], 1))

        # QUANT_BINS is an array: [NBINS]
        # q_r becomes:    [1 x 1 x NBINS]
        q_r = K.reshape(QUANT_BINS, (1, 1, NBINS))

        # get L1 distance from each element to each of the bins
        # dist is: [BATCH x WINDOW_SIZE x NBINS]
        dist = K.abs(x_r - q_r)

        # turn into softmax probabilities, which we return
        probs = softmax(self.SOFTMAX_TEMP * -dist)
        return probs
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], NBINS)

# dequantization: takes in    [BATCH x WINDOW_SIZE x NBINS]
#                 and returns [BATCH x WINDOW_SIZE]
class SoftmaxDequantization(Layer):
    def __init__(self, **kwargs):
        super(SoftmaxDequantization, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(SoftmaxDequantization, self).build(input_shape)
    
    def call(self, x, mask=None):
        out = K.dot(x, K.expand_dims(QUANT_BINS))
        out = K.reshape(out, (-1, out.shape[1]))
        return out
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])

# ---------------------------------------------------
# "Blocks" that make up all of our models
# ---------------------------------------------------

# activation used in all blocks
def activation(init = 0.3):
    # input is of form [nbatch x channel_size x num_channels],
    # so we share axis 1
    return PReLU(alpha_initializer = Constant(init),
                 shared_axes = [1])

# channel change block: takes input from however many channels
#                       it had before to [num_chans] channels,
#                       without applying any other operation
def channel_change_block(num_chans, filt_size):
    def f(inp):
        out = inp

        out = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear')(out)
        out = activation(0.3)(out)

        out = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear')(out)
        out = activation(0.3)(out)

        return out
    
    return f

# upsample block: takes input channels of length N and upsamples
#                 them to length 2N, using "phase shift" upsampling
def upsample_block(num_chans, filt_size):
    def f(inp):
        out = inp

        out = Conv1D(num_chans * 2, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear')(out)
        out = activation(0.3)(out)

        out = Conv1D(num_chans * 2, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear')(out)
        out = activation(0.3)(out)
        out = PhaseShiftUp1D(2)(out)

        return out
    
    return f

# downsample block: takes input channels of length N and downsamples
#                   them to length N/2, using strided convolution
def downsample_block(num_chans, filt_size):
    def f(inp):
        out = inp

        out = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear',
                     strides = 2)(out)
        out = activation(0.3)(out)

        out = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear')(out)
        out = activation(0.3)(out)

        return out
    
    return f

# advanced residual block, supporting gating and dilated convolutions
def residual_block(num_chans, filt_size, dilation = 1, gate = False,
                   operation = 'none'):
    def f(inp):
        # ---------------------------------------
        # shortcut connection
        # ---------------------------------------
        shortcut = inp

        # ---------------------------------------
        # residual operation
        # ---------------------------------------
        res = inp

        # conv1
        res = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear',
                     dilation_rate = dilation)(res)
        res = activation(0.3)(res)

        # conv2
        res = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear',
                     dilation_rate = dilation)(res)
        res = activation(0.3)(res)

        if (operation != 'none'):
            return res

        # ---------------------------------------
        # gating (if enabled)
        # ---------------------------------------
        if (gate):
            shortcut_gate = Conv1D(num_chans, 3, padding = 'same',
                                   kernel_initializer = W_INIT,
                                   bias_initializer = Constant(3),
                                   activation = 'sigmoid',
                                   dilation_rate = dilation)(inp)
            shortcut = Multiply()([shortcut, shortcut_gate])

            res_gate = Conv1D(num_chans, 3, padding = 'same',
                              kernel_initializer = W_INIT,
                              bias_initializer = Constant(3),
                              activation = 'sigmoid',
                              dilation_rate = dilation)(inp)
            res = Multiply()([res, res_gate])

        # ---------------------------------------
        # final output
        # ---------------------------------------
        return Add()([shortcut, res])
    
    return f




