# don't write bytecode for any file
import sys
sys.dont_write_bytecode = True

import scipy.signal as sig
import pywt
import numpy as np
import operator
import math

# parameters for sliding window, and window function (Hann)
STEP_SIZE = 160
OVERLAP_SIZE = 40
WINDOW_SIZE = STEP_SIZE + OVERLAP_SIZE
OVERLAP_FUNC = sig.hann(OVERLAP_SIZE * 2)

# directory that contains TIMIT files
TIMIT_DIR = "/home/sri/Desktop/timit"

# directory that contains .wav files to process
NUM_EPOCHS = 30
BATCH_SIZE = 64

# randomly shuffle data before partitioning into training/validation?
RANDOM_SHUFFLE = True

# sample rate of input file (used in MFCC calculation)
SAMPLE_RATE = 16000

# debug messages?
VERBOSE = True

# wavelet settings
WAVELET_TYPE = 'db6'
WAVELET_MODE = 'per'
WAVELET_LEVEL = 3

# calculate sizes of each decomposition level / size of the final output of
# the wavelet decomposition / where to split things (i.e. where in the full
# array one level ends, and the next one begins)
COEFF_SIZES = map(len, pywt.wavedec(np.zeros(WINDOW_SIZE), WAVELET_TYPE, WAVELET_MODE, WAVELET_LEVEL))
COEFF_SPLITS = [0] * (len(COEFF_SIZES) - 1)
for i in xrange(0, len(COEFF_SPLITS)):
	for j in xrange(0, i + 1):
		COEFF_SPLITS[i] += COEFF_SIZES[j]

print COEFF_SIZES
print COEFF_SPLITS

# total decomposition size
DECOMP_SIZE = COEFF_SIZES[-1] + COEFF_SPLITS[-1]




