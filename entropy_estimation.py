# ==========================================================================
# Parzen kernel-based entropy estimation
#     TODO: brief explanation and add citations
# ==========================================================================

import math
import numpy as np
from keras import backend as K
from consts import *

# the Parzen kernel is a zero-centered gaussian with bin-width standard deviation
std = (1.0 / (NBINS - 1))
norm = 1.0 / math.sqrt(2.0 * 3.14159 * std * std)
den = (2.0 * std * std)

def parzen_kernel(x):
    num = K.square(x)
    return norm * K.exp(-num / den)

# we use 10,000 samples to create our entropy estimate
N = 10000
log_2 = math.log(2.0)
bins = K.variable(np.linspace(-1.0, 1.0, NBINS))
r_bins = K.repeat_elements(bins.reshape((NBINS, 1)), N, 1)

def entropy_estimate(placeholder, code):
    # if there are less than N samples in this batch, we just use however much data
    # we have
    flt = K.flatten(code)
    end_idx = K.minimum(flt.shape[0], N)
    
    ref = flt[:end_idx]
    r_ref = K.repeat_elements(ref.reshape((1, end_idx)), NBINS, 0)

    r_kern = parzen_kernel(r_ref - r_bins[:, :end_idx])
    r_kern = K.sum(r_kern, axis = 1)
    r_kern /= K.sum(r_kern)

    ent = -K.sum(r_kern * K.log(r_kern + K.epsilon()) / log_2)
    
    return ent
