# don't write bytecode for any file
import sys
sys.dont_write_bytecode = True

import os
from numpy import round
from pylab import *
import numpy as np
from scipy.fftpack import idct, dct
import operator
import math
import scipy.io.wavfile as sciwav
import matplotlib.pyplot as plt
import pywt
from wavelets import *

from theano import tensor as T

from params import *
from utility import *

meanWin = 0
stdWin = 0


def computeMeanVariance(windows):
    global meanWin
    global stdWin

    meanWin = np.mean(windows, axis=0)
    stdWin = np.std(windows, axis=0)

    # replace zeros in STDs with very very small floats
    stdWin = np.where(stdWin == 0, np.finfo(float).eps, stdWin)


def processWindows(windows):
    dec = computeWaveletDecomp(windows)
    #dec = dec[:, :COEFF_SPLITS[0]]
    #dec = windows

    return dec

def normalizeWindows(windows):
    #Dmin = meanWin - (stdWin * 3)
    #Dmax = meanWin + (stdWin * 3)

    windows = (windows - meanWin) / (stdWin * 2)
    #windows = np.clip(windows, -1, 1)
    windows = np.tanh(windows)
    windows = np.clip(windows, -0.99999, 0.99999)

    return windows



def deprocessWindows(windows):
    #full = np.zeros((windows.shape[0], DECOMP_SIZE))
    #full[:, :COEFF_SPLITS[0]] = windows
    #rec = computeWaveletRecomp(full)

    rec = computeWaveletRecomp(windows)
    #rec = windows

    return rec

def denormalizeWindows(windows):
    #Dmin = meanWin - (stdWin * 3)
    #Dmax = meanWin + (stdWin * 3)

    windows = np.arctanh(windows)
    windows = (windows * (stdWin * 2)) + meanWin

    return windows

def denormalize(thing):
    return (thing * (stdWin * 2)) + meanWin


# (Theano) super advanced error function taking preprocessing and wavelet bands into account
def custom_error_function(y_true, y_pred):
    # clip to arctanh's range, then take arctanh and denormalize
    denorm_true = T.clip(y_true, -0.99999, 0.99999)
    denorm_true = T.arctanh(denorm_true)
    denorm_true = denormalize(denorm_true)

    denorm_pred = T.clip(y_pred, -0.99999, 0.99999)
    denorm_pred = T.arctanh(denorm_pred)
    denorm_pred = denormalize(denorm_pred)

    diff = denorm_true - denorm_pred

    # first 100 features (lower frequency bands) are easier to capture, and less emphasized than
    # the next 100 features (higher frequency bands)
    diff_b1 = diff[:, :25]
    diff_b2 = diff[:, 25:50]
    diff_b3 = diff[:, 50:100]
    diff_b4 = diff[:, 100:]
   
    diffSq_b1 = T.sqr(diff_b1)
    diffSq_b2 = T.sqr(diff_b2)
    diffSq_b3 = T.sqr(diff_b3)
    diffSq_b4 = T.sqr(diff_b4)

    error = 1 * T.mean(diffSq_b1) + 2 * T.mean(diffSq_b2) + 3 * T.mean(diffSq_b3) + 4 * T.mean(diffSq_b4) 
    
    return error





