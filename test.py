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
from keras.constraints import *

from windowingFunctions import *
from params import *
from utility import *
from mfcc import *
from preprocessing import *

# read in WAV files
print "Reading in .wav files..."
fileList = filesInDir(DATA_DIR)
rawWindows = []
for filepath in fileList:
    [rate, data] = sciwav.read(filepath)
    windows = extractWindows(data)

    if (rawWindows == []):
        rawWindows = windows
    else:
        rawWindows = np.append(rawWindows, windows, axis=0)

    #print filepath, ": ", windows.shape

# randomly shuffle data
if (RANDOM_SHUFFLE):
    rawWindows = np.random.permutation(rawWindows)

print "Raw windows shape: ", rawWindows.shape

# transform data and convert to float
print "Processing windows..."
processedWindows = processWindows(rawWindows)
processedWindows = processedWindows.astype(np.float32)

# compute mean and variance, then normalize by them
computeMeanVariance(processedWindows)
processedWindows = normalizeWindows(processedWindows)

print np.mean(np.abs(processedWindows), axis=None)
print np.std(np.abs(processedWindows), axis=None)


# 80/20 training/testing split
split = round(processedWindows.shape[0] * 0.8)

_train = (processedWindows[:split, :])
_test  = (processedWindows[split:, :])

autoencoder = Sequential()

#'''
autoencoder.add(Reshape(input_shape = (120,), dims=(120,)))
autoencoder.add(MaxoutDense(output_dim = 80, init = "glorot_normal", nb_feature = 15,
                            W_constraint = maxnorm(1)))
autoencoder.add(MaxoutDense(output_dim = 60, init = "glorot_normal", nb_feature = 15,
                            W_constraint = maxnorm(1)))
autoencoder.add(MaxoutDense(output_dim = 32, init = "glorot_normal", nb_feature = 10,
                            W_constraint = maxnorm(1)))
autoencoder.add(MaxoutDense(output_dim = 10, init = "glorot_normal", nb_feature = 5,
                            W_constraint = maxnorm(1)))
autoencoder.add(MaxoutDense(output_dim = 32, init = "glorot_normal", nb_feature = 5,
                            W_constraint = maxnorm(1)))
autoencoder.add(MaxoutDense(output_dim = 60, init = "glorot_normal", nb_feature = 10,
                            W_constraint = maxnorm(1)))
autoencoder.add(MaxoutDense(output_dim = 80, init = "glorot_normal", nb_feature = 15,
                            W_constraint = maxnorm(1)))
autoencoder.add(MaxoutDense(output_dim = 120, init = "glorot_normal", nb_feature = 15))
#'''

'''
autoencoder.add(Reshape(input_shape = (120,), dims=(120,)))
autoencoder.add(Dense(output_dim = 60, init = "glorot_normal", #nb_feature = 20,
                            W_constraint = maxnorm(2)))
autoencoder.add(PReLU())
#autoencoder.add(Activation("tanh"))
autoencoder.add(Dense(output_dim = 32, init = "glorot_normal", #nb_feature = 15,
                            W_constraint = maxnorm(2)))
autoencoder.add(PReLU())
#autoencoder.add(Activation("tanh"))
autoencoder.add(Dense(output_dim = 8, init = "glorot_normal", #nb_feature = 5,
                            W_constraint = maxnorm(2)))
autoencoder.add(PReLU())
#autoencoder.add(Activation("tanh"))
autoencoder.add(Dense(output_dim = 32, init = "glorot_normal", #nb_feature = 15,
                            W_constraint = maxnorm(2)))
autoencoder.add(PReLU())
#autoencoder.add(Activation("tanh"))
autoencoder.add(Dense(output_dim = 60, init = "glorot_normal", #nb_feature = 20,
                            W_constraint = maxnorm(2)))
autoencoder.add(PReLU())
#autoencoder.add(Activation("tanh"))
autoencoder.add(Dense(output_dim = 120, init = "glorot_normal", #nb_feature = 25,
                            W_constraint = maxnorm(2)))
'''

'''
autoencoder.add(Dense(input_shape = (160,), output_dim = 80, init = "glorot_normal",
                       activation='tanh'))
autoencoder.add(Dense(output_dim = 64, init = "glorot_normal", activation='tanh'))
autoencoder.add(Dense(output_dim = 40, init = "glorot_normal"))
autoencoder.add(GaussianDropout(0.1))
autoencoder.add(Dense(output_dim = 64, init = "glorot_normal", activation='tanh'))
autoencoder.add(Dense(output_dim = 80, init = "glorot_normal", activation='tanh'))
autoencoder.add(Dense(output_dim = 160, init = "glorot_normal", activation="tanh"))
#'''

# interesting observation: autoencoder going from raw => raw or dct => raw resembles
#     a bandpass filter
'''
autoencoder.add(Reshape(input_shape = (40,), dims = (1, 40, 1)))
autoencoder.add(Convolution2D(input_shape = (1, 40, 1), nb_filter = 32, nb_row = 10, nb_col = 1, init = "glorot_uniform",
                              border_mode = "same"))
autoencoder.add(PReLU())
autoencoder.add(Flatten(input_shape=(32, 40, 1)))
autoencoder.add(Dense(output_dim = 2048, init = "glorot_uniform"))
autoencoder.add(PReLU())
autoencoder.add(Dense(output_dim = 256, init = "glorot_uniform"))
autoencoder.add(PReLU())
autoencoder.add(Dense(output_dim = 15, init = "glorot_uniform"))
autoencoder.add(PReLU())
autoencoder.add(Dense(output_dim = 40, init = "glorot_uniform"))
autoencoder.add(PReLU())
autoencoder.add(Dense(output_dim = 40, init = "glorot_uniform", activation = "tanh"))
#autoencoder.add(Reshape(input_shape = (40,), dims = (1, 40, 1)))
#autoencoder.add(Convolution2D(input_shape = (1, 40, 1), nb_filter = 1, nb_row = 20, nb_col = 1, init = "glorot_uniform",
#                              border_mode = "same"))
#autoencoder.add(Reshape(input_shape = (1, 40, 1), dims = (40,)))
#'''

autoencoder.compile(loss = 'root_mean_squared_error', optimizer = Adam())

autoencoder.fit(_train, _train, nb_epoch = NUM_EPOCHS, batch_size = BATCH_SIZE,
		verbose = 1, validation_data = [_test, _test], show_accuracy = False)


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

    if replacement == True:
        nnReplaced = replaceInWindows(windows, autoencOutput)
        nnReplaced = reconstructFromWindows(nnReplaced)
        sciwav.write(prefix + "replaced.wav", rate, nnReplaced)

    print waveFilename, " mse: ", mse(nnReconstruction, desiredReconstruction)
    print waveFilename, " avg err: ", avgErr(nnReconstruction, desiredReconstruction)

autoencoderTest("./sp01.wav", "sp01", True)
autoencoderTest("./fiveYears.wav", "fy", True)
autoencoderTest("./sp19.wav", "sp19", True)




