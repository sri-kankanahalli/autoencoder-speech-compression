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

from load_data import *
from windowing import *

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
    # append random number to filename, so multiple simultaneous PESQ runs
    # don't interfere with each other (most likely...)
    #     TODO: turn "most likely" into "guarantee"
    random_suffix = str(random.randint(0, 99999999))
    clean_fname = "./temp/clean_" + random_suffix + ".wav"
    dirty_fname = "./temp/dirty_" + random_suffix + ".wav"

    # compute PESQ between original and corrupted waveforms
    sciwav.write(clean_fname, SAMPLE_RATE, clean_wav.astype(np.int16))
    sciwav.write(dirty_fname, SAMPLE_RATE, dirty_wav.astype(np.int16))
    pesq = run_pesq_filenames(clean_fname, dirty_fname)
    os.system("rm " + clean_fname)
    os.system("rm " + dirty_fname)
    
    return pesq

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





