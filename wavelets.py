from numpy import round
import numpy as np
import operator
import math
import pywt
from params import *

def iswt(coefficients, wavelet):
    """
      Input parameters: 

        coefficients
          approx and detail coefficients, arranged in level value 
          exactly as output from swt:
          e.g. [(cA1, cD1), (cA2, cD2), ..., (cAn, cDn)]

        wavelet
          Either the name of a wavelet or a Wavelet object

    """
    output = coefficients[0][0].copy() # Avoid modification of input data

    #num_levels, equivalent to the decomposition level, n
    num_levels = len(coefficients)
    for j in range(num_levels,0,-1): 
        step_size = int(math.pow(2, j-1))
        last_index = step_size
        _, cD = coefficients[num_levels - j]
        for first in range(last_index): # 0 to last_index - 1

            # Getting the indices that we will transform 
            indices = np.arange(first, len(cD), step_size)

            # select the even indices
            even_indices = indices[0::2] 
            # select the odd indices
            odd_indices = indices[1::2] 

            # perform the inverse dwt on the selected indices,
            # making sure to use periodic boundary conditions
            x1 = pywt.idwt(output[even_indices], cD[even_indices], wavelet, 'per') 
            x2 = pywt.idwt(output[odd_indices], cD[odd_indices], wavelet, 'per') 

            # perform a circular shift right
            x2 = np.roll(x2, 1)

            # average and insert into the correct indices
            output[indices] = (x1 + x2)/2.  

    return output




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
                print i, "/", numWindows
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
                print i, "/", numWindows
    return waveletRecomp


def computeSWTDecomp(windows):
    numWindows = windows.shape[0]
    waveletDecomp = np.zeros((numWindows, WINDOW_SIZE*4))

    i = 0
    for window in windows:
        w = pywt.swt(window, WAVELET_TYPE, 2)
        w = np.concatenate([w[0][0], w[0][1], w[1][0], w[1][1]])

        transformed = np.reshape(np.array(w), (1, WINDOW_SIZE*4))
        waveletDecomp[i, :] = transformed

        i += 1
    return waveletDecomp

def computeSWTRecomp(windows):
    numWindows = windows.shape[0]
    waveletRecomp = np.zeros((numWindows, WINDOW_SIZE))

    i = 0
    for window in windows:
        wl = np.split(window, [WINDOW_SIZE, WINDOW_SIZE*2, WINDOW_SIZE*3])
        wl = [[wl[0], wl[1]], [wl[2], wl[3]]]
        r = iswt(wl, WAVELET_TYPE)

        transformed = np.reshape(np.array(r), (1, WINDOW_SIZE))
        waveletRecomp[i, :] = transformed

        i += 1
    return waveletRecomp

        











