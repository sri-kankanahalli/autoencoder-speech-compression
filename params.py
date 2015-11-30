import scipy.signal as sig

# parameters for sliding window, and window function (Hann)
STEP_SIZE = 80
WINDOW_SIZE = STEP_SIZE * 2
WINDOW_FUNC = sig.hann(WINDOW_SIZE)

# directory that contains .wav files to process
DATA_DIR = "./entire/"

# randomly shuffle data before partitioning into training/validation?
RANDOM_SHUFFLE = False

# sample rate of input file (used in MFCC calculation)
SAMPLE_RATE = 8000

# number of MFCC coefficients
NUM_MFCC_COEFFS = 25

# debug messages?
VERBOSE = True
