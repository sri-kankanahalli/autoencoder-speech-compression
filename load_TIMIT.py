# ==========================================================================
# functions to load TIMIT .wavs and process the dataset
#     (mono, 16KHz, 32-bit)
# REQUIREMENT: you need to have run convert_TIMIT.py first
# ==========================================================================

import subprocess
import glob
import os
import scipy.io.wavfile as sciwav

from windowingFunctions import *
from params import *

def load_TIMIT_train():
    print "Reading in .wav files..."

    train_files_list = glob.glob(TIMIT_DIR + '/TIMIT/TRAIN/*/*/*.WAV')

    rawWindows = []
    i = 0
    for filepath in train_files_list:
        [rate, data] = sciwav.read(filepath)
        windows = extractWindows(data)
        windows = windows.tolist()

        if (rawWindows == []):
            rawWindows = windows
        else:
            rawWindows += windows

        print (str(i) + ": " + filepath + "\r"),
        i += 1
        if (i >= 1000): break

    rawWindows = np.array(rawWindows)
    rawWindows = rawWindows.astype(np.float32)

    print ""
    print "Raw windows shape: ", rawWindows.shape
    print np.amax(rawWindows)
    print np.amin(rawWindows)

    # v simple data augmentation
    rawWindowsX2 = np.clip(rawWindows * 2, -32767, 32767)
    rawWindowsD2 = np.clip(rawWindows / 2, -32767, 32767)
    #rawWindowsX3 = np.clip(rawWindows * 3, -32767, 32767)
    #rawWindowsD3 = np.clip(rawWindows / 3, -32767, 32767)

    augWindows = rawWindows
    augWindows = np.append(augWindows, rawWindowsX2, axis=0)
    augWindows = np.append(augWindows, rawWindowsD2, axis=0)
    #augWindows = np.append(augWindows, rawWindowsX3, axis=0)
    #augWindows = np.append(augWindows, rawWindowsD3, axis=0)

    print "Augmented windows shape: ", augWindows.shape

    # randomly shuffle data
    if (RANDOM_SHUFFLE):
        augWindows = np.random.permutation(augWindows)

    return augWindows
