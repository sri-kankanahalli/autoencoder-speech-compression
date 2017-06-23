from consts import *
from nn_util import *
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.initializers import *
from keras.activations import softmax

BLOCK_SIZE = 64
BLOCK_TRANSFORM = K.eye(BLOCK_SIZE)

# quantization: takes in    [BATCH x WINDOW_SIZE]
#               and returns [BATCH x WINDOW_SIZE x NBINS]
# where the last dimension is a one-hot vector of bins
#
# [bins initialization is in consts.py]
class BlockTransformQuantization(Layer):
    def build(self, input_shape):
        self.SOFTMAX_TEMP = K.variable(500.0, name = 'softmax_temp')
        self.trainable_weights = [QUANT_BINS] + \
                                 [self.SOFTMAX_TEMP] + \
                                 [BLOCK_TRANSFORM]
        super(BlockTransformQuantization, self).build(input_shape)
    
    # x is a vector: [BATCH_SIZE x BLOCK_SIZE]
    def step(self, x):
        # transform x by [QUANT_TRANSFORM]
        transformed_x = K.dot(x, BLOCK_TRANSFORM)
        
        # x_r becomes: [BATCH_SIZE x BLOCK_SIZE x 1]
        x_r = K.expand_dims(transformed_x, -1)

        # c_r becomes: [BATCH_SIZE x 1 x NBINS]
        c_r = K.expand_dims(QUANT_BINS, -2)

        # get L1 distance from each element to each of the bins
        # dist is: [BATCH_SIZE x BLOCK_SIZE x NBINS]
        dist = K.abs(x_r - c_r)

        # turn into softmax probabilities, which we return
        # quant is: [BATCH_SIZE x BLOCK_SIZE x NBINS]
        quant = softmax(self.SOFTMAX_TEMP * -dist)
        return quant
    
    def call(self, x, mask = None):        
        # out becomes: list of length [WINDOW_SIZE / BLOCK_SIZE]
        #                  of BATCH_SIZE x BLOCK_SIZE x NBINS length testors
        mod_x = K.reshape(x, (-1, x.shape[1] / BLOCK_SIZE, BLOCK_SIZE))
        mod_x = tf.unstack(tf.transpose(mod_x, [1, 0, 2]))
        out = []
        for i in mod_x:
            w = self.step(i)
            out.append(w)
        
        # we finagle this into: [BATCH_SIZE x WINDOW_SIZE x NBINS]
        enc = tf.transpose(tf.stack(out), [1, 0, 2, 3])
        enc = K.reshape(enc, (-1, enc.shape[1] * enc.shape[2], NBINS))

        quant_on = enc
        quant_off = K.zeros_like(enc)[:, :, 1:]
        quant_off = K.concatenate([K.reshape(x, (-1, x.shape[1], 1)),
                                   quant_off], axis = 2)
        return K.switch(QUANTIZATION_ON, quant_on, quant_off)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], NBINS)

# dequantization: takes in    [BATCH x WINDOW_SIZE x NBINS]
#                 and returns [BATCH x WINDOW_SIZE]
class BlockTransformDequantization(Layer):
    # x is a vector of size [BATCH_SIZE x BLOCK_SIZE x NBINS] -- 1 time step
    def step(self, x):
        quant = x
        
        # reconstructed window
        # d is: [BATCH_SIZE x BLOCK_SIZE x 1]
        d = K.dot(quant, K.expand_dims(QUANT_BINS, -1))
        recons = K.squeeze(d, -1)
        recons = K.dot(recons, tf.matrix_inverse(BLOCK_TRANSFORM))
        return recons
        
    def call(self, x, mask=None):
        mod_x = K.reshape(x, (-1, x.shape[1] / BLOCK_SIZE, BLOCK_SIZE, x.shape[2]))
        mod_x = tf.unstack(tf.transpose(mod_x, [1, 0, 2, 3]))
        out = []
        for i in mod_x:
            w = self.step(i)
            out.append(w)
        
        dec = tf.transpose(tf.stack(out), [1, 0, 2])
        dec = K.reshape(dec, (-1, dec.shape[1] * BLOCK_SIZE))

        quant_on = dec
        quant_off = K.reshape(x[:, :, :1], (-1, x.shape[1]))
        return K.switch(QUANTIZATION_ON, quant_on, quant_off)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])

