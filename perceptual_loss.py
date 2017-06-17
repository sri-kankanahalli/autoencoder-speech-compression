import numpy as np
import math
from numpy.fft import fft, ifft
from scipy.fftpack import dct, idct

from consts import *
from nn_util import *
from nn_blocks import *
from keras import backend as K

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

# given a (symbolic Keras) array of size M x A
#     this returns an array M x A where every one of the M samples has been independently
#     filtered by the DCT matrix passed in
def keras_dct(x, dct_mat):
    # reshape x into 2D array, and perform appropriate matrix operation
    reshaped_x = K.reshape(x, (-1, dct_mat.shape[0]))
    return K.dot(reshaped_x, dct_mat)

# ====================================================================
#  DFT (Discrete Fourier Transform)
# ====================================================================

# generate two square DFT matrices, one for the real component, one for
# the imaginary component
def generate_dft_mats(n):
    mat = np.fft.fft(np.eye(n))
    return K.variable(np.real(mat)), K.variable(np.imag(mat))

# given a (symbolic Keras) array of size M x WINDOW_SIZE
#     this returns an array M x WINDOW_SIZE where every one of the M samples has been replaced by
#     its DFT magnitude, using the DFT matrices passed in
def keras_dft_mag(x, real_mat, imag_mat):
    reshaped_x = K.reshape(x, (-1, real_mat.shape[0]))
    real = K.dot(reshaped_x, real_mat)
    imag = K.dot(reshaped_x, imag_mat)

    mag = K.sqrt(K.square(real) + K.square(imag) + K.epsilon())
    return mag

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

# ====================================================================
#  Finally: a simple perceptual loss function (based on Mel scale)
# ====================================================================

NUM_MFCC_COEFFS = 64

# precompute Mel filterbank
MEL_FILTERBANK_NPY = melFilterBank(NUM_MFCC_COEFFS).transpose()
MEL_FILTERBANK = K.variable(MEL_FILTERBANK_NPY)

# we precompute matrices for MFCC calculation
DFT_REAL, DFT_IMAG = generate_dft_mats(WINDOW_SIZE)
MFCC_DCT = generate_dct_mat(NUM_MFCC_COEFFS)

# given a (symbolic Keras) array of size M x WINDOW_SIZE
#     this returns an array M x N where each window has been replaced
#     by some perceptual transform (in this case, MFCC coeffs)
def perceptual_transform(x):
    powerSpectrum = K.square(keras_dft_mag(x, DFT_REAL, DFT_IMAG))
    filteredSpectrum = K.dot(powerSpectrum, MEL_FILTERBANK)
    logSpectrum = K.log(filteredSpectrum + K.epsilon())
    
    mfccs = keras_dct(logSpectrum, MFCC_DCT)[:, 1:-32]
    return mfccs

# perceptual loss function
def perceptual_distance(y_true, y_pred):
    y_true = K.reshape(y_true, (-1, WINDOW_SIZE))
    y_pred = K.reshape(y_pred, (-1, WINDOW_SIZE))
    
    pvec_true = perceptual_transform(y_true)
    pvec_pred = perceptual_transform(y_pred)
    
    return rmse(pvec_true, pvec_pred)





