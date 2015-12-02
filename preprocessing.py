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
from mfcc import *

meanTransformed = 0
stdTransformed = 0
meanOrig = 0
stdOrig = 0

def transformWindows(windows):
    #windows = dct(windows, type = 2, norm = 'ortho')
    return windows
    #return getMFCCsForWindows(windows)
    '''
    numWindows = windows.shape[0]

    transformedWindows = np.zeros((numWindows, 160))

    i = 0
    for window in windows:
        # preemphasize signal
        preemphasized = np.copy(window)
        for i in xrange(1, len(window)):
            preemphasized[i] = window[i] - 0.9 * window[i - 1]
        transformed = dct(preemphasized, type = 2, norm = 'ortho')
        #transformed = np.dot(transformed, FILTERBANK)
        #transformed = fft(window)
        #transformed = np.real(transformed)
        #transformed = np.concatenate([np.real(transformed), np.imag(transformed)], axis=0)
        transformed = np.reshape(np.array(transformed), (1, 160))

        transformedWindows[i, :] = transformed

        i += 1
        if (VERBOSE):
            if (i % 500 == 0):
                print i, "/", numWindows
    return transformedWindows
    #'''


def computeMeanVariance(transformed, orig):
    global meanTransformed
    global stdTransformed
    global meanOrig
    global stdOrig

    meanTransformed = np.mean(transformed, axis=0)
    stdTransformed = np.std(transformed, axis=0)
    meanOrig = np.mean(orig, axis=0)
    stdOrig = np.std(orig, axis=0)

    # replace zeros in STDs with very very small floats
    stdTransformed = np.where(stdTransformed == 0, np.finfo(float).eps, stdTransformed)
    stdOrig = np.where(stdOrig == 0, np.finfo(float).eps, stdOrig)

    #print "mean MFCC: ", meanTransformed
    #print "std MFCC: ", stdTransformed
    #print "mean window: ", meanOrig
    #print "std window: ", stdOrig



def preprocessTransformedWindows(windows):
    return windows

def preprocessOrigWindows(windows):
    if (PREPROC_METHOD == 'fft'):
        windows = fft(windows)
        windows = np.concatenate([np.real(windows), np.imag(windows)], axis=1)
    elif (PREPROC_METHOD == 'dct'):
        windows = dct(windows, type = 2, norm = 'ortho')  

    return windows

def normalizeTransformedWindows(windows):
    windows = (windows - meanTransformed) / stdTransformed
    windows = windows / 3.0
    windows = np.tanh(windows)
    return windows

def normalizeOrigWindows(windows):
    windows = (windows - meanOrig) / stdOrig
    windows = windows / 3.0
    windows = np.tanh(windows)
    return windows



def unpreprocessTransformedWindows(windows):
    return windows

def unpreprocessOrigWindows(windows):
    if (PREPROC_METHOD == 'fft'):
        real = windows[:, :WINDOW_SIZE]
        imag = windows[:, WINDOW_SIZE:]
    
        windows = real + (imag * 1j)
        windows = ifft(windows)
    elif (PREPROC_METHOD == 'dct'):
        windows = idct(windows, type = 2, norm = 'ortho')   
 
    return windows

def denormalizeTransformedWindows(windows):
    windows = np.arctanh(windows)
    windows = windows * 3.0
    windows = (windows * stdTransformed) + meanTransformed
    return windows

def denormalizeOrigWindows(windows):
    windows = np.arctanh(windows)
    windows = windows * 3.0
    windows = (windows * stdOrig) + meanOrig
    return windows








