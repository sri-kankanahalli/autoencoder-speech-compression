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

# ---------------------------------------------------
# Theano operation to quantize input into specified
# number of bins
# ---------------------------------------------------
class BinsQuantize(T.Op):
    # properties attribute
    __props__ = ()
    
    def __init__(self, nbins):
        self.nbins = nbins
        super(BinsQuantize, self).__init__()
        
    def make_node(self, x):
        assert hasattr(self, '_props'), "Your version of theano is too old to support __props__."
        x = T.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])
    
    def perform(self, node, inputs, output_storage):
        x, = inputs
        z, = output_storage
        
        s = (x + 1.0) / 2.0
        s = np.round(s * float(self.nbins - 1)) / float(self.nbins - 1)
        s = (s * 2.0) - 1.0
        
        z[0] = s
    
    def grad(self, input, output_gradients):
        # pass through gradients unchanged
        x, = input
        g, = output_gradients
        return [g]
        
    def infer_shape(self, node, i0_shapes):
        # output shape is same as input shape
        return i0_shapes


# ---------------------------------------------------
# "Phase shift" upsampling layer, as discussed in [that one
# superresolution paper]
#
# Takes vector of size: B x S  x nF
# And returns fector:   B x nS x F
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
        config = {'n' : self.n}
        base_config = super(PhaseShiftUp1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# ---------------------------------------------------
# Different types of "blocks" that make up all of our
# models
# ---------------------------------------------------

# weight initialization used in all blocks
W_INIT = 'he_uniform'

# activation used in all blocks
def activation():
    return LeakyReLU(0.3)

# residual block, going from NCHAN to NCHAN channels
def residual_block(num_chans, filt_size, dilation = 1):
    def f(input):
        shortcut = input
        
        res = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear',
                     use_bias = True,
                     dilation_rate = dilation)(input)
        res = activation()(res)
        res = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear',
                     use_bias = True,
                     dilation_rate = dilation)(res)
        
        m = Add()([shortcut, res])
        return m
    
    return f


# increase number of channels from 1 to NCHAN via convolution
def channel_increase_block(num_chans, filt_size):
    def f(input):
        shortcut = Permute((2, 1))(input)
        shortcut = UpSampling1D(num_chans)(shortcut)
        shortcut = Permute((2, 1))(shortcut)
        
        res = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear')(input)
        res = activation()(res)
        res = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear')(res)
        
        m = Add()([shortcut, res])
        return m
        
    return f


# downsample the signal 2x
def downsample_block(num_chans, filt_size):
    def f(input):
        shortcut = AveragePooling1D(2)(input)
        
        res = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear',
                     strides = 2)(input)
        res = activation()(res)
        res = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear')(res)
        
        m = Add()([shortcut, res])
        return m
    
    return f


# upsample the signal 2x
def upsample_block(num_chans, filt_size):
    def f(input):
        shortcut = UpSampling1D(2)(input)
        
        res = Conv1D(num_chans * 2, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear')(input)
        res = PhaseShiftUp1D(2)(res)
        res = activation()(res)
        res = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear')(res)
        
        m = Add()([shortcut, res])
        return m
    
    return f


# increase number of channels from NCHAN to 1 via convolution
def channel_decrease_block(num_chans, filt_size):
    def f(input):
        shortcut = Permute((2, 1))(input)
        shortcut = GlobalAveragePooling1D()(shortcut)
        shortcut = Reshape((-1, 1))(shortcut)
        
        res = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear')(input)
        res = activation()(res)
        res = Conv1D(1, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear')(res)

        m = Add()([shortcut, res])
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








