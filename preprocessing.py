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
from mfcc import *

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
    #dec = dec[:, COEFF_SPLITS[1]:COEFF_SPLITS[2]]

    dec = computeSWTDecomp(windows)
    dec = dec[:, :WINDOW_SIZE]
    return dec

def normalizeWindows(windows):
    windows = (windows - meanWin) / stdWin
    #windows = windows / 3.0
    #windows = np.tanh(windows)
    return windows



def deprocessWindows(windows):
    #full = np.zeros((windows.shape[0], DECOMP_SIZE))
    #full[:, COEFF_SPLITS[1]:COEFF_SPLITS[2]] = windows
    #rec = computeWaveletRecomp(full)
    full = np.zeros((windows.shape[0], WINDOW_SIZE*4))
    full[:, :WINDOW_SIZE] = windows
    rec = computeSWTRecomp(full)

    #windows[:, 0:160] = 0
    #windows[:, 320:480] = 0
    #rec = computeSWTRecomp(windows)
    return rec

def denormalizeWindows(windows):
    #windows = np.arctanh(windows)
    #windows = windows * 3.0
    windows = (windows * stdWin) + meanWin
    return windows


def replaceInWindows(origWindows, autoencOutput):
    processedWindows = computeSWTDecomp(origWindows)

    autoencOutput = denormalizeWindows(autoencOutput)
    processedWindows[:, :WINDOW_SIZE] = autoencOutput

    processedWindows = computeSWTRecomp(processedWindows)
    return processedWindows







