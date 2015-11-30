import scipy.signal as sig

# parameters for sliding window, and window function (Hann)
STEP_SIZE = 80
WINDOW_SIZE = STEP_SIZE * 2
WINDOW_FUNC = sig.hann(WINDOW_SIZE)
