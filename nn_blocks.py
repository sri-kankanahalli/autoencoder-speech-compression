# ==========================================================================
# neural network Keras layers / blocks / loss functions needed for model
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
W_INIT = 'he_normal'

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

# quantization: takes in    [BATCH x WINDOW_SIZE x 1]
#               and returns [BATCH x WINDOW_SIZE x NBINS]
# where the last dimension is a one-hot vector of bins
#
# [bins initialization is in consts.py]
class SoftmaxQuantization(Layer):
    def build(self, input_shape):
        self.SOFTMAX_TEMP = K.variable(500.0)
        self.trainable_weights = [QUANT_BINS,
                                  self.SOFTMAX_TEMP]
        super(SoftmaxQuantization, self).build(input_shape)
        
    def call(self, x, mask=None):
        # QUANT_BINS is an array: [NBINS]
        # q_r becomes:    [1 x 1 x NBINS]
        q_r = K.reshape(QUANT_BINS, (1, 1, -1))
        
        # get L1 distance from each element to each of the bins
        #     x is an array:   [BATCH x WINDOW_SIZE x 1]
        #     q_r is an array: [1 x 1 x NBINS]
        #     so dist is: [BATCH x WINDOW_SIZE x NBINS]
        dist = K.abs(x - q_r)
        
        # turn distances into soft bin assignments
        enc = softmax(self.SOFTMAX_TEMP * -dist)
        
        # if quantization is OFF, we just pass the input through unchanged,
        # in a hackish way
        quant_on = enc
        quant_off = K.concatenate([x,
                                   K.zeros_like(enc)[:, :, 1:]], axis = 2)
        
        return K.switch(QUANTIZATION_ON, quant_on, quant_off)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], NBINS)

# dequantization: takes in    [BATCH x WINDOW_SIZE x NBINS]
#                 and returns [BATCH x WINDOW_SIZE x 1]
class SoftmaxDequantization(Layer):
    def call(self, x, mask=None):
        dec = K.sum(x * QUANT_BINS, axis = -1)
        dec = K.reshape(dec, (-1, dec.shape[1], 1))

        quant_on = dec
        quant_off = x[:, :, :1]
        return K.switch(QUANTIZATION_ON, quant_on, quant_off)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1)

# ---------------------------------------------------
# "Blocks" that make up all of our models
# ---------------------------------------------------

# activation used in all blocks
def activation(init = 0.3):
    # input is of form [NBATCH x CHANNEL_SIZE x NUM_CHANNELS],
    # so we share axis 1
    return PReLU(alpha_initializer = Constant(init),
                 shared_axes = [1])


# channel change block: takes input from however many channels
#                       it had before to [num_chans] channels
def channel_change_block(num_chans, filt_size):
    def f(inp):
        shortcut = inp
        res = inp

        shortcut = Conv1D(num_chans, filt_size, padding = 'same',
                          kernel_initializer = W_INIT,
                          activation = 'linear')(shortcut)
        shortcut = activation(0.3)(shortcut)

        res = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear')(res)
        res = activation(0.3)(res)

        res = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear')(res)
        res = activation(0.3)(res)

        return Add()([shortcut, res])
    
    return f

# upsample block: takes input channels of length N and upsamples
#                 them to length 2N, using "phase shift" upsampling
def upsample_block(num_chans, filt_size):
    def f(inp):
        shortcut = inp
        res = inp

        shortcut = Conv1D(num_chans * 2, filt_size, padding = 'same',
                          kernel_initializer = W_INIT,
                          activation = 'linear')(shortcut)
        shortcut = activation(0.3)(shortcut)
        shortcut = PhaseShiftUp1D(2)(shortcut)

        res = Conv1D(num_chans * 2, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear')(res)
        res = activation(0.3)(res)
        res = PhaseShiftUp1D(2)(res)

        res = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear')(res)
        res = activation(0.3)(res)

        return Add()([shortcut, res])
    
    return f

# downsample block: takes input channels of length N and downsamples
#                   them to length N/2, using strided convolution
def downsample_block(num_chans, filt_size):
    def f(inp):
        shortcut = inp
        res = inp

        shortcut = Conv1D(num_chans, filt_size, padding = 'same',
                          kernel_initializer = W_INIT,
                          activation = 'linear',
                          strides = 2)(shortcut)
        shortcut = activation(0.3)(shortcut)

        res = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear',
                     strides = 2)(res)
        res = activation(0.3)(res)

        res = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear')(res)
        res = activation(0.3)(res)

        return Add()([shortcut, res])
    
    return f

# residual block
def residual_block(num_chans, filt_size, dilation = 1):
    def f(inp):
        shortcut = inp
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

        return Add()([shortcut, res])
    
    return f


# ---------------------------------------------------
# Loss functions
# ---------------------------------------------------

# entropy weight variable
tau = K.variable(0.0, name = "entropy_weight")

# NaN-safe RMSE loss function
def rmse(y_true, y_pred):
    mse = K.mean(K.square(y_pred - y_true), axis=-1)
    return K.sqrt(mse + K.epsilon())

def code_entropy(placeholder, code):
    # [BATCH_SIZE x NBINS]
    #     => [NBINS]
    # probability distribution over symbols for each channel
    all_onehots = K.reshape(code, (-1,  NBINS))
    onehot_hist = K.sum(all_onehots, axis = 0)
    onehot_hist /= K.sum(onehot_hist)

    # compute entropy of probability distribution
    entropy = -K.sum(onehot_hist * K.log(onehot_hist + K.epsilon()) / K.log(2.0))
    loss = tau * entropy
    return K.switch(QUANTIZATION_ON, loss, K.zeros_like(loss))

def code_sparsity(placeholder, code):
    # [BATCH_SIZE x CHANNEL_SIZE x NBINS]
    #     => [BATCH_SIZE x CHANNEL_SIZE]
    sqrt_sum = K.sum(K.sqrt(code + K.epsilon()), axis = -1) - 1.0
    
    # take mean over each soft assignment
    sparsity = K.mean(sqrt_sum, axis = -1)
    return K.switch(QUANTIZATION_ON, sparsity, K.zeros_like(sparsity))



















