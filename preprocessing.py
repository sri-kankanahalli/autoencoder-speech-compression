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
import pywt
from wavelets import *

from theano import tensor as T
from theano_dct import *

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
    #dec = computeWaveletDecomp(windows)
    #dec = dec[:, :COEFF_SPLITS[0]]
    dec = windows

    return dec

def normalizeWindows(windows):
    #Dmin = meanWin - (stdWin * 3)
    #Dmax = meanWin + (stdWin * 3)

    #windows = (windows - meanWin) / (stdWin * 2)
    #windows = np.clip(windows, -1, 1)
    #windows = np.tanh(windows)
    #windows = np.clip(windows, -0.99999, 0.99999)
    windows = windows / 32768

    return windows



def deprocessWindows(windows):
    #full = np.zeros((windows.shape[0], DECOMP_SIZE))
    #full[:, :COEFF_SPLITS[0]] = windows
    #rec = computeWaveletRecomp(full)

    #rec = computeWaveletRecomp(windows)
    rec = windows

    return rec

def denormalizeWindows(windows):
    #Dmin = meanWin - (stdWin * 3)
    #Dmax = meanWin + (stdWin * 3)

    #windows = np.arctanh(windows)
    #windows = (windows * (stdWin * 2)) + meanWin
    windows = windows * 32768

    return windows

def denormalize(thing):
    return (thing * (stdWin * 2)) + meanWin


# (Theano) super advanced error function taking preprocessing and wavelet bands into account
def custom_error_function(y_true, y_pred):
    # compute MSE over frequency domain
    dct_true = theano_dct(y_true)
    dct_pred = theano_dct(y_pred)

    diff = dct_true - dct_pred
    error = T.mean(T.sqr(diff))
    
    return error





