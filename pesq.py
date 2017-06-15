# ==========================================================================
# utility functions for PESQ (Perceptual Evaluation of Speech Quality)
# 
# REQUIREMENT: you need to have compiled PESQ and put it in the current
#              folder. we used the -Ofast optimization flag in GCC
# ==========================================================================

import re
import os
import scipy.io.wavfile as sciwav
import numpy as np
import random
import ctypes

from load_data import *
from windowing import *

pesq_dll = ctypes.CDLL('./PESQ.so')
pesq_dll.pesq.restype = ctypes.c_double

# interface to PESQ evaluation, taking in two filenames as input
def run_pesq_filenames(clean, to_eval):
    pesq_regex = re.compile("\(MOS-LQO\):  = ([0-9]+\.[0-9]+)")
    
    pesq_out = os.popen("./PESQ +16000 +wb " + clean + " " + to_eval).read()
    regex_result = pesq_regex.search(pesq_out)
    
    if (regex_result is None):
        return 0.0
    else:
        return float(regex_result.group(1))

# interface to PESQ evaluation, taking in two waveforms as input
def run_pesq_waveforms(clean_wav, dirty_wav):
    clean_wav = clean_wav.astype(np.double)
    dirty_wav = dirty_wav.astype(np.double)

    return pesq_dll.pesq(ctypes.c_void_p(clean_wav.ctypes.data),
                         ctypes.c_void_p(dirty_wav.ctypes.data),
                         len(clean_wav),
                         len(dirty_wav))

# interface to PESQ evaluation, taking in two sets of windows as input
def run_pesq_windows(clean_wnd, dirty_wnd, wparam1, wparam2):
    clean_wnd = np.reshape(clean_wnd, (-1, WINDOW_SIZE))
    clean_wav = reconstruct_from_windows(clean_wnd, OVERLAP_SIZE, OVERLAP_FUNC)
    clean_wav = unpreprocess_waveform(clean_wav, wparam1)
    clean_wav = np.clip(clean_wav, -32767, 32767)

    dirty_wnd = np.reshape(dirty_wnd, (-1, WINDOW_SIZE))
    dirty_wav = reconstruct_from_windows(dirty_wnd, OVERLAP_SIZE, OVERLAP_FUNC)
    dirty_wav = unpreprocess_waveform(dirty_wav, wparam2)
    dirty_wav = np.clip(dirty_wav, -32767, 32767)
    
    return run_pesq_waveforms(clean_wav, dirty_wav)

# scales PESQ output from MOS-LQO [1.0, 4.5ish] to [0, 1]
def scale_pesq(pesq):
    out = (pesq - 1.0) / (4.5 - 1.0)
    return np.clip(out, 0.0, 1.0)





