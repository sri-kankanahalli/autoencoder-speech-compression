from numpy import round
import numpy as np
import operator
import math
import pywt
from params import *


def computeWaveletDecomp(windows):
    numWindows = windows.shape[0]
    waveletDecomp = np.zeros((numWindows, DECOMP_SIZE))

    i = 0
    for window in windows:
        w = pywt.wavedec(window, WAVELET_TYPE, WAVELET_MODE, WAVELET_LEVEL)
        w = [val for subl in w for val in subl]
        transformed = np.reshape(np.array(w), (1, DECOMP_SIZE))

        waveletDecomp[i, :] = transformed

        i += 1
        if (VERBOSE):
            if (i % 2000 == 0):
                print (str(i) + " / " + str(numWindows))
    return waveletDecomp



def computeWaveletRecomp(windows):
    numWindows = windows.shape[0]
    waveletRecomp = np.zeros((numWindows, WINDOW_SIZE))

    i = 0
    for window in windows:
        wl = np.split(window, COEFF_SPLITS)
        r = pywt.waverec(wl, WAVELET_TYPE, WAVELET_MODE)
        transformed = np.reshape(np.array(r), (1, WINDOW_SIZE))

        waveletRecomp[i, :] = transformed

        i += 1
        if (VERBOSE):
            if (i % 2000 == 0):
                print (str(i) + " / " + str(numWindows))
    return waveletRecomp













