# ==========================================================================
# base parameters that won't change
# ==========================================================================

import scipy.signal as sig

# parameters for sliding window, and window function (Hann)
WINDOW_SIZE = 512
OVERLAP_SIZE = 32
STEP_SIZE = WINDOW_SIZE - OVERLAP_SIZE
OVERLAP_FUNC = sig.hann(OVERLAP_SIZE * 2)

# sample rate of input files
SAMPLE_RATE = 16000

# directory that contains TIMIT files
TIMIT_DIR = "/home/sri/Desktop/timit"

# weight initialization used in all layers of network
W_INIT = 'glorot_normal'

