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
allWindows = []
for filepath in fileList:
    [rate, data] = sciwav.read(filepath)
    windows = extractWindows(data)

    if (allWindows == []):
        allWindows = windows
    else:
        allWindows = np.append(allWindows, windows, axis=0)

    #print filepath, ": ", windows.shape

# randomly shuffle data
if (RANDOM_SHUFFLE):
    allWindows = np.random.permutation(allWindows)
print "All windows shape: ", allWindows.shape

# get MFCCs for windows
print "Calculating MFCCs..."
allMFCCs = getMFCCsForWindows(allWindows)

# convert to float
allWindows = allWindows.astype(np.float32)
allMFCCs   = allMFCCs.astype(np.float32)





# preprocessing
allMFCCs = preprocessMFCCs(allMFCCs)
allWindows = preprocessWindows(allWindows)

computeMeanVariance(allMFCCs, allWindows)

allMFCCs = normalizeMFCCs(allMFCCs)
allWindows = normalizeWindows(allWindows)

# training/testing split
split = round(allMFCCs.shape[0] * 0.9)

X_train = (allMFCCs[:split, :])
Y_train = (allWindows[:split, :])
X_test  = (allMFCCs[split:, :])
Y_test  = (allWindows[split:, :])

autoencoder = Sequential()
autoencoder.add(GaussianDropout(input_shape = (NUM_MFCC_COEFFS,), p = 0.05))
autoencoder.add(Dense(output_dim = 1024, init = "glorot_uniform", activation = "relu"))
autoencoder.add(Dense(output_dim = 512, init = "glorot_uniform", activation = "relu"))
autoencoder.add(Dense(output_dim = 300, init = "glorot_uniform", activation = "relu"))
autoencoder.add(Dense(output_dim = 240, init = "glorot_uniform", activation = "relu"))
autoencoder.add(Dense(output_dim = 160, init = "glorot_uniform"))

autoencoder.compile(loss = 'root_mean_squared_error', optimizer = RMSprop())

autoencoder.fit(X_train, Y_train, nb_epoch = 500, batch_size = 64,
		verbose = 1, validation_data = [X_test, Y_test], show_accuracy = False)


[rate, data] = sciwav.read("sp01.wav")
windows = extractWindows(data)

# first, write desired reconstruction
desiredReconstruction = reconstructFromWindows(windows)
sciwav.write("desired.wav", rate, desiredReconstruction)
    
# then, run NN on MFCCs
mfccs = getMFCCsForWindows(windows)
mfccs = preprocessMFCCs(mfccs)
mfccs = normalizeMFCCs(mfccs)

predicted = autoencoder.predict(mfccs, batch_size = 64, verbose = 1)
predicted = denormalizeWindows(predicted)
predicted = unpreprocessWindows(predicted)

nnReconstruction = reconstructFromWindows(predicted)
sciwav.write("nnOutput.wav", rate, nnReconstruction)



'''
[rate, data] = sciwav.read("fiveYears.wav")
print data.shape

windows = extractWindows(data)
print windows.shape

reconstruction = reconstructFromWindows(windows)
print reconstruction.shape

sciwav.write("out.wav", rate, reconstruction)

r = reconstruction[:data.shape[0]]
print "mse: ", mse(r, data)
print "avg err: ", avgErr(r, data)

'''
