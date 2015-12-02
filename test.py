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

from windowingFunctions import *
from params import *
from utility import *
from mfcc import *
from preprocessing import *

# read in WAV files
print "Reading in .wav files..."
fileList = filesInDir(DATA_DIR)
origWindows = []
for filepath in fileList:
    [rate, data] = sciwav.read(filepath)
    windows = extractWindows(data)

    if (origWindows == []):
        origWindows = windows
    else:
        origWindows = np.append(origWindows, windows, axis=0)

    #print filepath, ": ", windows.shape

# randomly shuffle data
if (RANDOM_SHUFFLE):
    origWindows = np.random.permutation(origWindows)
print "Original windows shape: ", origWindows.shape

# get MFCCs for windows
print "Calculating window transformation..."
transformedWindows = transformWindows(origWindows)

# convert to float
origWindows = origWindows.astype(np.float32)
transformedWindows = transformedWindows.astype(np.float32)





# preprocessing
transformedWindows = preprocessTransformedWindows(transformedWindows)
origWindows = preprocessOrigWindows(origWindows)

# compute mean and variance, then normalize by them
computeMeanVariance(transformedWindows, origWindows)
transformedWindows = normalizeTransformedWindows(transformedWindows)
origWindows = normalizeOrigWindows(origWindows)

print np.mean(np.abs(transformedWindows), axis=None)
print np.std(np.abs(transformedWindows), axis=None)
print np.mean(np.abs(origWindows), axis=None)
print np.std(np.abs(origWindows), axis=None)

# 80/20 training/testing split
split = round(transformedWindows.shape[0] * 0.8)

X_train = (transformedWindows[:split, :])
Y_train = (origWindows[:split, :])
X_test  = (transformedWindows[split:, :])
Y_test  = (origWindows[split:, :])

autoencoder = Sequential()

'''
autoencoder.add(Dense(input_shape = (25,), output_dim = 100, init = "glorot_normal"))
autoencoder.add(Dense(output_dim = 160, init = "glorot_normal", activation = "tanh"))
autoencoder.add(Dense(input_shape = (160,), output_dim = 25600, init = "glorot_normal"))
autoencoder.add(Dense(output_dim = 160, init = "glorot_normal"))
#'''

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
#'''
autoencoder.add(Reshape(input_shape = (160,), dims = (1, 160, 1)))
autoencoder.add(Convolution2D(input_shape = (1, 160, 1), nb_filter = 64, nb_row = 16, nb_col = 1, init = "glorot_uniform",
                              border_mode = "same"))
autoencoder.add(Flatten(input_shape=(64, 160, 1)))
autoencoder.add(Dense(output_dim = 1024, init = "glorot_uniform", activation = "tanh"))
autoencoder.add(Dense(output_dim = 256, init = "glorot_uniform"))
autoencoder.add(Dense(output_dim = 40, init = "glorot_uniform", activation = "tanh"))
autoencoder.add(Dense(output_dim = 160, init = "glorot_uniform"))
autoencoder.add(Reshape(input_shape = (160,), dims = (1, 160, 1)))
autoencoder.add(Convolution2D(input_shape = (1, 160, 1), nb_filter = 1, nb_row = 64, nb_col = 1, init = "glorot_uniform",
                              activation = "tanh", border_mode = "same"))
autoencoder.add(Reshape(input_shape = (1, 160, 1), dims = (160,)))
#'''

autoencoder.compile(loss = 'root_mean_squared_error', optimizer = RMSprop())

autoencoder.fit(X_train, Y_train, nb_epoch = 100, batch_size = 64,
		verbose = 1, validation_data = [X_test, Y_test], show_accuracy = False)


def autoencoderTest(waveFilename, desiredFilename, reconstructionFilename):
    [rate, data] = sciwav.read(waveFilename)
    windows = extractWindows(data)

    # first, write desired reconstruction
    desiredReconstruction = reconstructFromWindows(windows)
    sciwav.write(desiredFilename, rate, desiredReconstruction)
    
    # then, run NN on transformed windows
    transformed = transformWindows(windows)
    transformed = preprocessTransformedWindows(transformed)
    transformed = normalizeTransformedWindows(transformed)

    predicted = autoencoder.predict(transformed, batch_size = 64, verbose = 1)
    predicted = denormalizeOrigWindows(predicted)
    predicted = unpreprocessOrigWindows(predicted)

    nnReconstruction = reconstructFromWindows(predicted)
    sciwav.write(reconstructionFilename, rate, nnReconstruction)

autoencoderTest("./sp01.wav", "sp01desired.wav", "sp01output.wav")
autoencoderTest("./fiveYears.wav", "fydesired.wav", "fyoutput.wav")
autoencoderTest("./sp19.wav", "sp19desired.wav", "sp19output.wav")




