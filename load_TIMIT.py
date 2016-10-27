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

def load_TIMIT_train(timitDir, numLoad = -1):
    print "Reading in .wav files..."

    train_files_list = glob.glob(timitDir + '/TIMIT/TRAIN/*/*/*.WAV')

    rawData = []
    i = 0
    for filepath in train_files_list:
        [rate, data] = sciwav.read(filepath)
        data = data.astype(np.float64)

        if (rawData == []):
            rawData = [data]
        else:
            rawData += [data]

        print (str(i) + ": " + filepath + "\r"),
        i += 1

        if (numLoad > 0):
            if (i == numLoad): break
        

    return rawData


