import numpy as np
from numpy.fft import fft
from scipy.fftpack import idct, dct
import math
import matplotlib.pyplot as plt
from params import *

# based on a combination of this article:
#     http://practicalcryptography.com/miscellaneous/machine-learning/...
#         guide-mel-frequency-cepstral-coefficients-mfccs/
# and some of this code:
#     http://stackoverflow.com/questions/5835568/...
#         how-to-get-mfcc-from-an-fft-on-a-signal

def freqToMel(freq):
    return 1127.01048 * math.log(1 + freq / 700.0)

def melToFreq(mel):
    return 700 * (math.exp(mel / 1127.01048) - 1)

def melFilterBank(numCoeffs):
    minHz = 0
    maxHz = SAMPLE_RATE / 2            # by Nyquist theorem
    numFFTBins = WINDOW_SIZE

    maxMel = freqToMel(maxHz)
    minMel = freqToMel(minHz)

    # we need (numCoeffs + 2) points to create (numCoeffs) filterbanks
    melRange = np.array(xrange(numCoeffs + 2))
    melRange = melRange.astype(np.float32)

    # create (numCoeffs + 2) points evenly spaced between minMel and maxMel
    melCenterFilters = melRange * (maxMel - minMel) / (numCoeffs + 1) + minMel

    
    for i in xrange(numCoeffs + 2):
        # mel domain => frequency domain
        melCenterFilters[i] = melToFreq(melCenterFilters[i])

        # frequency domain => FFT bins
        melCenterFilters[i] = math.floor(numFFTBins * melCenterFilters[i] / maxHz)       

    # create matrix of filters (one row is one filter)
    filterMat = np.zeros((numCoeffs, numFFTBins))

    # generate filters (in frequency domain) and plot
    for i in range(1, numCoeffs + 1):
        filter = np.zeros(numFFTBins)
        
        startRange = melCenterFilters[i - 1]
        midRange   = melCenterFilters[i]
        endRange   = melCenterFilters[i + 1]

        for j in range(startRange, midRange):
            filter[j] = (float(j) - startRange) / (midRange - startRange)
        for j in range(midRange, endRange):
            filter[j] = 1 - ((float(j) - midRange) / (endRange - midRange))
        
        filterMat[i - 1] = filter
        #plt.plot(filter)
    print melCenterFilters
    #plt.show()

    # return filterbank as matrix
    return filterMat

# precomputed Mel filterbank
#     (transpose so we can do dot products with the power spectrum)
FILTERBANK = melFilterBank(NUM_MFCC_COEFFS).transpose()

# compute MFCC for single window
def mfcc(signal):
    # preemphasize signal
    #preemphasizedSignal = np.copy(signal)
    #for i in xrange(1, len(signal)):
    #    preemphasizedSignal[i] = signal[i] - 0.9 * signal[i - 1]

    complexSpectrum = fft(signal)
    powerSpectrum = abs(complexSpectrum) ** 2
    filteredSpectrum = np.dot(powerSpectrum, FILTERBANK)

    # replace places where filtered spectrum is zero
    filteredSpectrum = np.where(filteredSpectrum == 0, np.finfo(float).eps, \
                                filteredSpectrum)

    # get log spectrum and take DCT to get MFCC
    logSpectrum = np.log(filteredSpectrum)
    mfcc = dct(logSpectrum, type=2)

    return mfcc

# compute MFCC for list of windows
def getMFCCsForWindows(windows):
    numWindows = windows.shape[0]

    mfccs = np.zeros((numWindows, NUM_MFCC_COEFFS))

    i = 0
    for window in windows:
        windowMFCC = mfcc(window)
        windowMFCC = np.reshape(np.array(windowMFCC), (1, len(windowMFCC)))

        mfccs[i, :] = windowMFCC

        i += 1
        if (VERBOSE):
            if (i % 500 == 0):
                print i, "/", numWindows
    return mfccs




