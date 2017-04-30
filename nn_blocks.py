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
from entropy_estimation import entropy_estimate

# ---------------------------------------------------
# Theano operation to quantize input into specified
# number of bins
# ---------------------------------------------------
class Quantize(T.Op):
    # properties attribute
    __props__ = ()
    
    def __init__(self, nbins):
        self.nbins = nbins
        super(Quantize, self).__init__()
        
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

def QuantizeLayer(nbins):
    return Lambda(lambda x : Quantize(nbins)(x),
                  output_shape = lambda s : s)


# ---------------------------------------------------
# Theano operation to stochastically quantize input into specified
# number of bins
# ---------------------------------------------------
class StochasticQuantize(T.Op):
    # properties attribute
    __props__ = ()
    
    def __init__(self, nbins):
        self.nbins = nbins
        super(StochasticQuantize, self).__init__()
        
    def make_node(self, x):
        assert hasattr(self, '_props'), "Your version of theano is too old to support __props__."
        x = T.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])
    
    def perform(self, node, inputs, output_storage):
        x, = inputs
        z, = output_storage
        
        # transform from [-1, 1] to [0, 1]
        s = (x + 1.0) / 2.0

        # from [0, 1] to [0, NBINS - 1] (continuous)
        s = s * float(self.nbins - 1)

        # get decimals (which correspond to rounding probabilities)
        prob_thresh = np.mod(s, 1.0)

        # generate random values in [0, 1) and use those to make rounding
        # decisions
        probs = np.random.random_sample(s.shape)
        rnd = np.greater(prob_thresh, probs)

        # from [0, NBINS - 1] (continuous) to [0, NBINS - 1] (discrete)
        s = np.floor(s) + rnd

        # from [0, BINS - 1] to [0, 1]
        s = s / float(self.nbins - 1)

        # from [0, 1] to [-1, 1]
        s = (s * 2.0) - 1.0
        
        z[0] = s
    
    def grad(self, input, output_gradients):
        # pass through gradients unchanged (since the expected value of
        # rounding is just x)
        x, = input
        g, = output_gradients
        return [g]
        
    def infer_shape(self, node, i0_shapes):
        # output shape is same as input shape
        return i0_shapes


class StochasticQuantizeLayer(Layer):
    def __init__(self, nbins, **kwargs):
        super(StochasticQuantizeLayer, self).__init__(**kwargs)
        self.nbins = nbins
        self.uses_learning_phase = True
    
    def build(self, input_shape):
        self.train_op = StochasticQuantize(self.nbins)
        self.test_op = Quantize(self.nbins)
        self.trainable_weights = []
    
    def call(self, x, mask=None):
        x = K.in_train_phase(self.train_op(x), self.test_op(x))
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

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
                     activation = 'linear',
                     use_bias = True,
                     dilation_rate = dilation)(input)
        res = activation()(res)
        res = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear',
                     use_bias = True,
                     dilation_rate = dilation)(res)
        res = activation()(res)
        
        m = Add()([shortcut, res])
        return m
    
    return f


# increase number of channels from from_chan to num_chans via convolution
def channel_increase_block(num_chans, filt_size, from_chan = 1):
    def f(input):
        if (num_chans % from_chan != 0):
            raise ValueError('num_chans must be divisible by from_chan')

        shortcut = Permute((2, 1))(input)
        shortcut = UpSampling1D(num_chans / from_chan)(shortcut)
        shortcut = Permute((2, 1))(shortcut)
        
        res = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear')(input)
        res = activation()(res)
        res = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear')(res)
        res = activation()(res)
        
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
        res = activation()(res)
        
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
        res = activation()(res)
        
        m = Add()([shortcut, res])
        return m
    
    return f


# increase number of channels from num_chans to to_chan via convolution
def channel_decrease_block(num_chans, filt_size, to_chan = 1):
    def f(input):
        shortcut = Permute((2, 1))(input)
        shortcut = GlobalAveragePooling1D()(shortcut)
        shortcut = Reshape((-1, 1))(shortcut)
        if (to_chan > 1):
            shortcut = Permute((2, 1))(shortcut)
            shortcut = UpSampling1D(to_chan)(shortcut)
            shortcut = Permute((2, 1))(shortcut)
        
        res = Conv1D(num_chans, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear')(input)
        res = activation()(res)
        res = Conv1D(to_chan, filt_size, padding = 'same',
                     kernel_initializer = W_INIT,
                     activation = 'linear')(res)
        res = activation()(res)

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

# euclidean distance "layer"
def EuclideanDistance():
    def func(vects):
        x, y = vects
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True) + K.epsilon())

    def shape(shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    return Lambda(func, output_shape = shape)

# map for load_model
KERAS_LOAD_MAP = {'PhaseShiftUp1D' : PhaseShiftUp1D,
                  'Quantize' : Quantize,
                  'StochasticQuantize' : StochasticQuantize,
                  'StochasticQuantizeLayer' : StochasticQuantizeLayer,
                  'NBINS' : NBINS,
                  'entropy_estimate' : entropy_estimate,
                  'rmse' : rmse,
                  'EuclideanDistance': EuclideanDistance}





