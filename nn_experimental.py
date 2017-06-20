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

ADAPT_STEP = 32
CHANGE_SCALES = K.variable(np.random.uniform(0.9, 1.1, (NBINS, 1)))

# quantization: takes in    [BATCH x WINDOW_SIZE]
#               and returns [BATCH x WINDOW_SIZE x NBINS]
# where the last dimension is a one-hot vector of bins
#
# [bins initialization is in consts.py]
class DeltaQuantization(Layer):
    def build(self, input_shape):
        self.SOFTMAX_TEMP = K.variable(500.0, name = 'softmax_temp')
        self.trainable_weights = [QUANT_BINS,
                                  CHANGE_SCALES,
                                  self.SOFTMAX_TEMP]
        super(DeltaQuantization, self).build(input_shape)
    
    # x is a vector: [BATCH_SIZE x ADAPT_STEP]
    # curr_pred is a vector: [BATCH_SIZE x 1]
    # curr_bins is a vector: [BATCH_SIZE x NBINS]
    def step(self, x, curr_pred, curr_bins):
        # delta is: difference from what's in [ADAPT_STEP] window
        #           and actual prediction
        #               [BATCH_SIZE x ADAPT_STEP]
        delta = x - curr_pred
        
        # d_r becomes: [BATCH_SIZE x ADAPT_STEP x 1]
        d_r = K.expand_dims(delta, -1)

        # c_r becomes: [BATCH_SIZE x 1 x NBINS]
        c_r = K.expand_dims(curr_bins, -2)

        # get L1 distance from each element to each of the bins
        # dist is: [BATCH_SIZE x ADAPT_STEP x NBINS]
        dist = K.abs(d_r - c_r)

        # turn into softmax probabilities, which we return
        # quant is: [BATCH_SIZE x ADAPT_STEP x NBINS]
        quant = softmax(self.SOFTMAX_TEMP * -dist)
        
        # update current prediction with mean of (reconstructed) deltas
        #     d becomes: [BATCH_SIZE x ADAPT_STEP]
        d = K.batch_dot(quant, K.expand_dims(curr_bins, -1))
        d = K.squeeze(d, -1)
        new_pred = curr_pred + K.expand_dims(K.mean(d, axis = 1), 1)
        
        # FINALLY: update bins

        # symbol_probs is: [BATCH_SIZE x NBINS]
        symbol_probs = K.sum(quant, axis = 1) / ADAPT_STEP
               
        # curr_change_scale is: [BATCH_SIZE x 1]
        curr_change_scale = K.dot(symbol_probs, CHANGE_SCALES)
        
        # new_bins is: [BATCH_SIZE x NBINS x 1], then [BATCH_SIZE x NBINS]
        new_bins = K.batch_dot(K.expand_dims(curr_bins), K.expand_dims(curr_change_scale))
        new_bins = K.squeeze(new_bins, -1)
        
        return quant, new_pred, new_bins
    
    def call(self, x, mask = None):
        # predictions always starts at zero
        curr_pred = tf.zeros([tf.shape(x)[0], 1])
        
        # bins always starts at QUANT_BINS
        curr_bins = K.expand_dims(QUANT_BINS, 0)
        tile_amt = [tf.shape(x)[0]]
        tile_amt = tf.concat([tile_amt, tf.ones((1,), dtype = tf.int32)], axis = 0)
        curr_bins = tf.tile(curr_bins, tile_amt)
        
        # out becomes: list of length [WINDOW_SIZE / ADAPT_STEP]
        #                  of BATCH_SIZE x ADAPT_STEP x NBINS length testors
        mod_x = K.reshape(x, (-1, x.shape[1] / ADAPT_STEP, ADAPT_STEP))
        mod_x = tf.unstack(tf.transpose(mod_x, [1, 0, 2]))
        out = []
        for i in mod_x:
            w, curr_pred, curr_bins = self.step(i, curr_pred, curr_bins)
            out.append(w)
        
        # we finagle this into: [BATCH_SIZE x WINDOW_SIZE x NBINS]
        enc = tf.transpose(tf.stack(out), [1, 0, 2, 3])
        enc = K.reshape(enc, (-1, enc.shape[1] * enc.shape[2], NBINS))
        return enc
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], NBINS)

# dequantization: takes in    [BATCH x WINDOW_SIZE x NBINS]
#                 and returns [BATCH x WINDOW_SIZE]
class DeltaDequantization(Layer):
    # x is a vector of size [BATCH_SIZE x ADAPT_STEP x NBINS] -- 1 time step
    # curr_pred is a vector: [BATCH_SIZE x 1]
    # curr_bins is a vector: [BATCH_SIZE x NBINS]
    def step(self, x, curr_pred, curr_bins):
        quant = x
        
        d = K.batch_dot(quant, K.expand_dims(curr_bins, -1))
        d = K.squeeze(d, -1)
        out = curr_pred + d
        
        # update current prediction with mean of deltas
        new_pred = curr_pred + K.expand_dims(K.mean(d, axis = 1), 1)
        
        # FINALLY: update bins

        # symbol_probs is: [BATCH_SIZE x NBINS]
        symbol_probs = K.sum(quant, axis = 1) / ADAPT_STEP
        
        # curr_change_scale is: [BATCH_SIZE x 1]
        curr_change_scale = K.dot(symbol_probs, CHANGE_SCALES)
        
        # new_bins is: [BATCH_SIZE x NBINS x 1], then [BATCH_SIZE x NBINS]
        new_bins = K.batch_dot(K.expand_dims(curr_bins), K.expand_dims(curr_change_scale))
        new_bins = K.squeeze(new_bins, -1)
        
        return out, new_pred, new_bins
        
    def call(self, x, mask=None):
        # predictions always start at zero
        curr_pred = tf.zeros([tf.shape(x)[0], 1])
        
        # bins always starts at QUANT_BINS
        curr_bins = K.expand_dims(QUANT_BINS, 0)
        tile_amt = [tf.shape(x)[0]]
        tile_amt = tf.concat([tile_amt, tf.ones((1,), dtype = tf.int32)], axis = 0)
        curr_bins = tf.tile(curr_bins, tile_amt)
        
        mod_x = K.reshape(x, (-1, x.shape[1] / ADAPT_STEP, ADAPT_STEP, x.shape[2]))
        mod_x = tf.unstack(tf.transpose(mod_x, [1, 0, 2, 3]))
        out = []
        for i in mod_x:
            w, curr_pred, curr_bins = self.step(i, curr_pred, curr_bins)
            out.append(w)
            
        dec = tf.transpose(tf.stack(out), [1, 0, 2])
        dec = K.reshape(dec, (-1, dec.shape[1] * ADAPT_STEP))
        return dec
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])
