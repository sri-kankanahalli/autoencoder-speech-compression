# ==========================================================================
# base parameters that won't change
# ==========================================================================

import scipy.signal as sig
import numpy as np

# number of threads to use in any CPU multithreaded code
NUM_THREADS = 8

# standard batch size
BATCH_SIZE = 128

# number of speech files for train, val, and test
TRAIN_SIZE = 2000
VAL_SIZE = 100
TEST_SIZE = 500

# randomly shuffle data before partitioning into training/validation?
RANDOM_SHUFFLE = True

# during training, we evaluate PESQ and RMSE and such on full speech files every epoch, which
# is kind of expensive. so instead of selecting the full training and validation set, we
# randomly select this many waveforms
TRAIN_EVALUATE = 100
VAL_EVALUATE = 100

# parameters for sliding window, and window function (triangular)
WINDOW_SIZE = 512
OVERLAP_SIZE = 32
STEP_SIZE = WINDOW_SIZE - OVERLAP_SIZE
OVERLAP_FUNC = sig.hann(OVERLAP_SIZE * 2)
WINDOWING_MULT = np.concatenate([OVERLAP_FUNC[:OVERLAP_SIZE],
                                 np.ones(WINDOW_SIZE - OVERLAP_SIZE * 2),
                                 OVERLAP_FUNC[OVERLAP_SIZE:]])

# extract mode:
#     0 - apply overlap function during window extraction
#     1 - apply overlap function during reconstruction
EXTRACT_MODE = 0

# sample rate of input files
SAMPLE_RATE = 16000

# directory that contains TIMIT files
TIMIT_DIR = "/home/sri/Desktop/timit"

# number of quantization bins
NBINS = 32


