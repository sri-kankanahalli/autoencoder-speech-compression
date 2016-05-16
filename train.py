# don't write bytecode for any file
import sys
sys.dont_write_bytecode = True

import os
from numpy import round
from pylab import *
import numpy as np
import operator
import math
import scipy.io.wavfile as sciwav
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import containers
from keras.layers.noise import *
from keras.optimizers import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.regularizers import *
from keras.layers.advanced_activations import *
from keras.layers.normalization import *
from keras.layers.recurrent import *
from keras.constraints import *

from windowingFunctions import *
from params import *
from utility import *
from preprocessing import *
from load_TIMIT import *

# read in TIMIT training set
rawWindows = load_TIMIT_train()

# transform data and convert to float
print "Processing windows..."
processedWindows = processWindows(rawWindows)
processedWindows = processedWindows.astype(np.float32)

# compute mean and variance, then normalize
computeMeanVariance(processedWindows)
processedWindows = normalizeWindows(processedWindows)

print processedWindows.shape

print np.mean(np.abs(processedWindows), axis=None)
print np.std(np.abs(processedWindows), axis=None)

# 80/20 training/testing split
print processedWindows.shape
split = int(round(processedWindows.shape[0] * 0.8))

_train = (processedWindows[:split, :])
_test  = (processedWindows[split:, :])

autoencoder = Sequential()
autoencoder.add(Reshape( (200,), input_shape = (200,)))

autoencoder.add(Dense(output_dim = 256, init = "glorot_normal", activation='relu'))
#autoencoder.add(Dropout(0.05))
autoencoder.add(Dense(output_dim = 128, init = "glorot_normal", activation='relu'))
#autoencoder.add(Dropout(0.05))
autoencoder.add(Dense(output_dim = 64, init = "glorot_normal", activation='relu'))
#autoencoder.add(Dropout(0.05))
autoencoder.add(Dense(output_dim = 48, init = "glorot_normal", activation='relu'))
#autoencoder.add(Dropout(0.05))

autoencoder.add(Dense(output_dim = 32, init = "glorot_normal", activation='tanh',
                      activity_regularizer = activity_l1(10e-5) ))

autoencoder.add(Dense(output_dim = 48, init = "glorot_normal", activation='relu'))
#autoencoder.add(Dropout(0.05))
autoencoder.add(Dense(output_dim = 64, init = "glorot_normal", activation='relu'))
#autoencoder.add(Dropout(0.05))
autoencoder.add(Dense(output_dim = 128, init = "glorot_normal", activation='relu'))
#autoencoder.add(Dropout(0.05))
autoencoder.add(Dense(output_dim = 256, init = "glorot_normal", activation='relu'))
#autoencoder.add(Dropout(0.05))

autoencoder.add(Dense(output_dim = 200, init = "glorot_normal", activation='tanh'))

autoencoder.compile(loss = custom_error_function, optimizer = Adam())

autoencoder.fit(_train, _train, nb_epoch = NUM_EPOCHS, batch_size = BATCH_SIZE,
		verbose = 1, validation_data = [_test, _test], show_accuracy = True, shuffle=True)

'''
autoencoder.add(Reshape( (25, 1), input_shape = (25,)))
autoencoder.add(Convolution1D(input_dim = 1,
                              input_length = 25,
                              nb_filter = 32,
                              filter_length = 5,
                              border_mode = "same",
                              activation = "relu"))
autoencoder.add(Dropout(0.1))
autoencoder.add(Flatten())
'''

def autoencoderTest(waveFilename, prefix, replacement=False):
    [rate, data] = sciwav.read(waveFilename)
    windows = extractWindows(data)

    # first, write desired reconstruction
    desiredWindows = processWindows(windows)
    desiredWindows = normalizeWindows(desiredWindows)
    desiredWindows = denormalizeWindows(desiredWindows)
    desiredWindows = deprocessWindows(desiredWindows)
    desiredReconstruction = reconstructFromWindows(desiredWindows)
    sciwav.write(prefix + "desired.wav", rate, desiredReconstruction)
    
    # then, run NN on transformed windows
    transformed = processWindows(windows)
    transformed = normalizeWindows(transformed)

    autoencOutput = autoencoder.predict(transformed, batch_size = 64, verbose = 1)
    predicted = denormalizeWindows(autoencOutput)
    predicted = deprocessWindows(predicted)

    nnReconstruction = reconstructFromWindows(predicted)
    sciwav.write(prefix + "output.wav", rate, nnReconstruction)

    #if replacement == True:
    #    nnReplaced = replaceInWindows(windows, autoencOutput)
    #    nnReplaced = reconstructFromWindows(nnReplaced)
    #    sciwav.write(prefix + "replaced.wav", rate, nnReplaced)

    print waveFilename, " mse: ", mse(nnReconstruction, desiredReconstruction)
    print waveFilename, " avg err: ", avgErr(nnReconstruction, desiredReconstruction)

# SA1 is already trained; fiveYears and SX383 are not
autoencoderTest("./SA1.WAV", "SA1", True)
autoencoderTest("./fiveYears.wav", "fy", True)
autoencoderTest("./SX383.WAV", "SX383", True)




