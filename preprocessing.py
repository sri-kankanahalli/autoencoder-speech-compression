# don't write bytecode for any file
import sys
sys.dont_write_bytecode = True

import os
from numpy import round
from pylab import *
import numpy as np
import operator
import math
import scipy.io.wavfile as sciwav
import matplotlib.pyplot as plt

from theano import tensor as T
from theano_dct import *

from params import *
from utility import *


def processWindows(windows):
    # all non-normalization-related preprocessing steps go here
    #     (there are none currently)
    return windows

def normalizeWindows(windows):
    windows = windows / 32768

    return windows



def deprocessWindows(windows):
    # all non-normalization-related un-preprocessing steps go here
    return windows

def denormalizeWindows(windows):
    windows = windows * 32768

    return windows


# (Theano) super advanced error function 
def custom_error_function(y_true, y_pred):
    # transfer signals from time to frequency domain
    dft_true = theano_dft(y_true)
    dft_pred = theano_dft(y_pred)

    # compute difference in time and frequency domains
    freq_diff = dft_true - dft_pred
    time_diff = y_true - y_pred

    # MSE in time and freq domain
    mse_freq = T.mean(T.sqr(freq_diff))
    mse_time = T.mean(T.sqr(time_diff))

    # error is magnitude of (Rmse_freq, Rmse_time)
    error = T.sqrt(mse_freq + mse_time)
    
    return error





