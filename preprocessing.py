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
    #dec = dec[:, COEFF_SPLITS[1]:COEFF_SPLITS[2]]

    return dec

def normalizeWindows(windows):
    windows = (windows - meanWin) / stdWin
    #windows = windows / 3.0
    #windows = np.tanh(windows)
    return windows



def deprocessWindows(windows):
    #full = np.zeros((windows.shape[0], DECOMP_SIZE))
    #full[:, COEFF_SPLITS[1]:COEFF_SPLITS[2]] = windows
    rec = computeWaveletRecomp(windows)
    #full = np.zeros((windows.shape[0], WINDOW_SIZE*4))
    #full[:, :WINDOW_SIZE] = windows

    return rec

def denormalizeWindows(windows):
    #windows = np.arctanh(windows)
    #windows = windows * 3.0
    windows = (windows * stdWin) + meanWin
    return windows








