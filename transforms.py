import numpy as np
import theano.tensor as T
import theano
import math
from numpy.fft import fft, ifft
from scipy.fftpack import dct, idct
from keras import backend as K

from consts import *

# ====================================================================
#  DCT (Discrete Cosine Transform)
# ====================================================================

# generate square dct matrix
#     how to use: generate n-by-n matrix M. then, if you have a signal w, then:
#                 dct(w) = M * w
#     where w must be n-by-1
#
#     backed by scipy
def generate_dct_mat(n, norm = 'ortho'):
    return K.variable(dct(np.eye(n), norm = norm))

# given a (symbolic Theano) array of size M x A
#     this returns an array M x A where every one of the M samples has been independently
#     filtered by the DCT matrix passed in
def theano_dct(x, dct_mat):
    import theano.tensor as T

    # reshape x into 2D array, and perform appropriate matrix operation
    reshaped_x = x.reshape((1, x.shape[0], x.shape[1]))

    result = T.tensordot(dct_mat, reshaped_x, [[0], [2]])
    result = result.reshape((result.shape[0], result.shape[2])).T

    return result

# ====================================================================
#  DFT (Discrete Fourier Transform)
# ====================================================================

# generate two square DFT matrices, one for the real component, one for
# the imaginary component
def generate_dft_mats(n):
    mat = np.fft.fft(np.eye(n))
    return K.variable(np.real(mat)), K.variable(np.imag(mat))

# given a (symbolic Theano) array of size M x WINDOW_SIZE [or M x WINDOW_SIZE x 1]
#     this returns an array M x WINDOW_SIZE where every one of the M samples has been replaced by
#     its DFT magnitude, using the DFT matrices passed in
def theano_dft_mag(x, real_mat, imag_mat):
    import theano.tensor as T

    reshaped_x = x.reshape((1, x.shape[0], x.shape[1]))

    real = T.tensordot(real_mat, reshaped_x, [[0], [2]])
    real = real.reshape((real.shape[0], real.shape[2])).T
    
    imag = T.tensordot(imag_mat, reshaped_x, [[0], [2]])
    imag = imag.reshape((imag.shape[0], imag.shape[2])).T

    result = T.sqrt(T.sqr(real) + T.sqr(imag) + K.epsilon())

    return result

# ====================================================================
#  MFCC (Mel Frequency Cepstral Coefficients)
# ====================================================================

# based on a combination of this article:
#     http://practicalcryptography.com/miscellaneous/machine-learning/...
#         guide-mel-frequency-cepstral-coefficients-mfccs/
# and some of this code:
#     http://stackoverflow.com/questions/5835568/...
#         how-to-get-mfcc-from-an-fft-on-a-signal

# conversions between Mel scale and regular frequency scale
def freqToMel(freq):
    return 1127.01048 * math.log(1 + freq / 700.0)

def melToFreq(mel):
    return 700 * (math.exp(mel / 1127.01048) - 1)

# generate Mel filter bank
def melFilterBank(numCoeffs):
    minHz = 0
    maxHz = SAMPLE_RATE / 2            # max Hz by Nyquist theorem
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

    # generate triangular filters (in frequency domain)
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
 

    # return filterbank as matrix
    return filterMat




