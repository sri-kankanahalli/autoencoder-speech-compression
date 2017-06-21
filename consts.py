# ==========================================================================
# base parameters that won't change
# ==========================================================================

import scipy.signal as sig
import numpy as np
from keras import backend as K

# number of speech files for train, val, and test
TRAIN_SIZE = 1000
VAL_SIZE = 100
TEST_SIZE = 500

# randomly shuffle data before partitioning into training/validation?
RANDOM_SHUFFLE = True

# during training, we evaluate PESQ and RMSE and such on full speech files every epoch, which
# is kind of expensive. so instead of selecting the full training and validation set, we
# randomly select this many waveforms
TRAIN_EVALUATE = 100
VAL_EVALUATE = 100

# parameters for sliding window, and window function (Hann)
WINDOW_SIZE = 512
OVERLAP_SIZE = 32
STEP_SIZE = WINDOW_SIZE - OVERLAP_SIZE
OVERLAP_FUNC = sig.triang(OVERLAP_SIZE * 2)

# sample rate of input files
SAMPLE_RATE = 16000

# directory that contains TIMIT files
TIMIT_DIR = "/home/sri/Desktop/timit"

# number of quantization bins, as well as initialization
NBINS = 31
BINS_INIT = np.linspace(-1.0, 1.0, NBINS)
QUANT_BINS = K.variable(BINS_INIT, name = 'QUANT_BINS')

# quantization is initially turned off
QUANTIZATION_ON = K.variable(False, name = 'QUANTIZATION_ON')


