import os
from numpy import round
from pylab import *
import numpy as np
from scipy.fftpack import idct, dct
import operator
import math
import scipy.io.wavfile as sciwav
import matplotlib.pyplot as plt

from params import *
from utility import *

meanMFCC = 0
stdMFCC = 0
meanWindow = 0
stdWindow = 0

def computeMeanVariance(mfccs, windows):
    global meanMFCC
    global stdMFCC
    global meanWindow
    global stdWindow

    meanMFCC = np.mean(mfccs, axis=0)
    stdMFCC = np.std(mfccs, axis=0)
    meanWindow = np.mean(windows, axis=0)
    stdWindow = np.std(windows, axis=0)

    print "mean MFCC: ", meanMFCC
    print "std MFCC: ", stdMFCC
    print "mean window: ", meanWindow
    print "std window: ", stdWindow

def preprocessMFCCs(mfccs):
    return mfccs

def preprocessWindows(windows):
    windows = dct(windows, type = 2, norm = 'ortho')
    return windows

def normalizeMFCCs(mfccs):
    mfccs = (mfccs - meanMFCC) / stdMFCC
    return mfccs

def normalizeWindows(windows):
    windows = (windows - meanWindow) / stdWindow
    return windows



def unpreprocessMFCCs(mfccs):
    return mfccs

def unpreprocessWindows(windows):
    windows = idct(windows, type = 2, norm = 'ortho')
    return windows

def denormalizeMFCCs(mfccs):
    mfccs = (mfccs * stdMFCC) + meanMFCC
    return mfccs

def denormalizeWindows(windows):
    windows = (windows * stdWindow) + meanWindow
    return windows
