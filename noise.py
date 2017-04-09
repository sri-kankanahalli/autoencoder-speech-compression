# ==========================================================================
# types of noise
# ==========================================================================

from scipy.fftpack import dct, idct
import numpy as np

from consts import *

# different types of noise
def identity(window, param):
    return window

def additive_noise(window, param):
    corrupted = np.copy(window)
    corrupted += np.random.uniform(-param, param, corrupted.shape)
    corrupted = np.clip(corrupted, -1.0, 1.0)
    return corrupted

def mult_noise(window, param):
    corrupted = np.copy(window)
    corrupted *= np.random.normal(1.0, param, corrupted.shape)
    corrupted = np.clip(corrupted, -1.0, 1.0)
    return corrupted
    
def high_freq_additive_noise(window, param):
    crange = WINDOW_SIZE / 2
    
    corrupted = np.copy(window)
    corrupted = dct(corrupted, norm = 'ortho')
    corrupted[:, crange:] += np.random.uniform(-param, param, (crange,))
    corrupted = idct(corrupted, norm = 'ortho')
    corrupted = np.clip(corrupted, -1.0, 1.0)
    return corrupted

def low_freq_additive_noise(window, param):
    crange = WINDOW_SIZE / 2
    
    corrupted = np.copy(window)
    corrupted = dct(corrupted, norm = 'ortho')
    corrupted[:, :crange] += np.random.uniform(-param, param, (crange,))
    corrupted = idct(corrupted, norm = 'ortho')
    corrupted = np.clip(corrupted, -1.0, 1.0)
    return corrupted

# list of noise functions, and parameters for each
noise_types = [
               (identity,
                   [None]),
               (additive_noise,
                   [1.0 / 1024, 1.0 / 256, 1.0 / 64]),
               (mult_noise,
                   [1.0 / 64, 1.0 / 16, 1.0 / 8]),
               (high_freq_additive_noise,
                   [1.0 / 512, 1.0 / 128, 1.0 / 32]),
               (low_freq_additive_noise,
                   [1.0 / 512, 1.0 / 128, 1.0 / 32])
              ]
