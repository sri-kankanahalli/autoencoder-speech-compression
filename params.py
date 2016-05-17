# don't write bytecode for any file
import sys
sys.dont_write_bytecode = True

import scipy.signal as sig
import numpy as np
import operator
import math

# parameters for sliding window, and window function (Hann)
STEP_SIZE = 280
OVERLAP_SIZE = 40
WINDOW_SIZE = STEP_SIZE + OVERLAP_SIZE
OVERLAP_FUNC = sig.hann(OVERLAP_SIZE * 2)

# directory that contains TIMIT files
TIMIT_DIR = "/home/sri/Desktop/timit"

# directory that contains .wav files to process
NUM_EPOCHS = 1
BATCH_SIZE = 64

# randomly shuffle data before partitioning into training/validation?
RANDOM_SHUFFLE = True

# sample rate of input file (used in MFCC calculation)
SAMPLE_RATE = 16000



