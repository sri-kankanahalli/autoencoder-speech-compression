# ==========================================================================
# functions to load TIMIT .wavs, make train/val/test splits, and process
# the dataset into speech windows
#     (all WAV files are mono, 16KHz, 16-bit)
# REQUIREMENT: you need to have run convert_TIMIT.py first
# ==========================================================================

import subprocess
import glob
import os
import scipy.io.wavfile as sciwav
import numpy as np
import random

from windowing import *
from consts import *

# ---------------------------------------------------
# directory walk functions
# ---------------------------------------------------
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def get_wavs_contained(a_dir):
    return [name for name in os.listdir(a_dir)
            if not os.path.isdir(os.path.join(a_dir, name))
               and name.split(".")[-1].lower() == 'wav']

# ---------------------------------------------------
# load info about TIMIT from directory/file structure
# ---------------------------------------------------

# number of dialects in TIMIT (no reason this should be anything
# other than 8, but it's here). when making our train/val/test
# split, we try to achieve as equal a distribution over the dialects
# as possible
NUM_DIALECTS = 8

# extract info about the structure of TIMIT into a global variable
# (there's no reason to have more than one TIMIT corpus at a time)
#     structure: {split (TRAIN/TEST) =>
#                     [dialect (0-7) =>
#                         {speaker ID => [list of WAVs]}
#                     ]
#                }
TIMIT = {'TRAIN' : [], 'TEST' : []}
for i in xrange(1, NUM_DIALECTS + 1):
    for split in ['TRAIN', 'TEST']:
        dir_name = TIMIT_DIR + '/TIMIT/' + split + '/DR' + str(i) + '/'
        speakers = get_immediate_subdirectories(dir_name)
        
        info = {}
        for name in speakers:
            info[name] = sorted(get_wavs_contained(dir_name + name))

        TIMIT[split].append(info)

# ---------------------------------------------------
# create train/test/validation splits on TIMIT
# ---------------------------------------------------
# generates full path given a split, dialect, speaker, and filename
def generate_timit_filepath(split, dialect, speaker, wav):
    return TIMIT_DIR + '/TIMIT/' + split + '/DR' + str(dialect) + '/' + \
                        speaker + '/' + wav

# generates a "split" on the TIMIT dataset -- a certain number of file
# paths from the TRAIN/TEST that already exists, equally proportioned among
# the different dialects (as closely as possible), with the option of
# separating the speakers for training vs. validation
def generate_timit_set(split, num_wavs, speaker_range = None, seed = 1337):
    random.seed(seed)
    
    # form a list of tuples (dialect, speaker, wav) for every speech file for
    # every speaker for every dialect
    dialect_queue = []
    for dialect in xrange(0, NUM_DIALECTS):
        # we can use only a certain range of speakers for each dialect, to force
        # training and validation to have disjoint speaker sets if we want
        speakers_for_dialect = sorted(TIMIT[split][dialect].keys())
        if (speaker_range is not None):
            lower_pct = speaker_range[0]
            upper_pct = speaker_range[1]
            
            lower_idx = int(len(speakers_for_dialect) * lower_pct)
            upper_idx = int(len(speakers_for_dialect) * upper_pct)
            
            speakers_for_dialect = speakers_for_dialect[lower_idx:upper_idx]

        to_add = []

        for speaker in speakers_for_dialect:
            wavs = TIMIT[split][dialect][speaker]

            for wav in wavs:
                tup = (dialect, speaker, wav)
                to_add.append(tup)

        # randomly shuffle the output
        random.shuffle(to_add)
        dialect_queue.append(to_add)

    # distribute the tuples equally to an output list, using a LIFO queue system
    out = []
    for i in xrange(0, num_wavs):
        dialect_tups = dialect_queue.pop(0)
        while (not dialect_tups):
            dialect_tups = dialect_queue.pop(0)
        tup = dialect_tups.pop(0)

        out.append(tup)
        if (dialect_tups):
            dialect_queue.append(dialect_tups)
        if (not dialect_queue):
            break

    # turn list of tuples into list of filepaths
    out = [generate_timit_filepath(split, x[0] + 1, x[1], x[2]) for x in out]
    return out

# generates train, test, and validation splits on TIMIT filepaths, ensuring the
# following properties:
#     - we use as close to an equal distribution over dialects as possible
#     - there are no speakers overlapping between train, test, and validation
#
# returns 3 lists of filepaths
def timit_train_test_val(num_train, num_val, num_test, val_spkr_pct = 0.25):
    train = generate_timit_set('TRAIN', num_train, (0.0, 1.0 - val_spkr_pct))
    val = generate_timit_set('TRAIN', num_val, (1.0 - val_spkr_pct, 1.0))
    test = generate_timit_set('TEST', num_test)
    
    return train, val, test

# ---------------------------------------------------
# load raw waveforms from a list
# ---------------------------------------------------
def load_raw_waveforms(lst):
    rawData = []
    #i = 0
    for filepath in lst:
        [rate, data] = sciwav.read(filepath)
        data = data.astype(np.float64)

        if (rawData == []):
            rawData = [data]
        else:
            rawData += [data]
        
        #print (str(i) + ": " + filepath + "\r"),
        #i += 1
    
    return rawData

# ---------------------------------------------------
# waveform preprocessing functions
# ---------------------------------------------------
def preprocess_waveform(waveform):
    # scale waveform between -1 and 1 (maximizing its volume)
    mn = np.min(waveform)
    mx = np.max(waveform)
    maxabs = np.maximum(np.abs(mn), np.abs(mx))
        
    return np.copy(waveform) / maxabs, (maxabs,)

def unpreprocess_waveform(waveform, params):
    return np.copy(waveform) * params[0]

# ---------------------------------------------------
# load and process TIMIT data into speech windows
# ---------------------------------------------------
def load_data(num_train, num_val, num_test,
              pre_func = None):
    if (pre_func is None):
        pre_func = preprocess_waveform

    # generate train/val/test split paths
    train_paths, val_paths, test_paths = \
        timit_train_test_val(num_train, num_val, num_test)

    # raw waveforms
    train_waveforms = load_raw_waveforms(train_paths)
    val_waveforms = load_raw_waveforms(val_paths)
    test_waveforms = load_raw_waveforms(test_paths)

    # waveform preprocessing in action
    train_procwave = np.copy(train_waveforms)
    val_procwave = np.copy(val_waveforms)
    test_procwave = np.copy(test_waveforms)

    train_wparams = [()] * len(train_procwave)
    val_wparams = [()] * len(val_procwave)
    test_wparams = [()] * len(test_procwave)

    # preprocess every waveform
    for i in xrange(0, len(train_procwave)):
        train_procwave[i], train_wparams[i] = \
            pre_func(train_procwave[i])
    for i in xrange(0, len(val_procwave)):
        val_procwave[i], val_wparams[i] = \
            pre_func(val_procwave[i])
    for i in xrange(0, len(test_procwave)):
        test_procwave[i], test_wparams[i] = \
            pre_func(test_procwave[i])

    # turn each waveform into a corresponding list of windows
    train_windows = extract_windows_multiple(train_procwave, STEP_SIZE,
                                             OVERLAP_SIZE, collapse = False)
    val_windows = extract_windows_multiple(val_procwave, STEP_SIZE,
                                           OVERLAP_SIZE, collapse = False)
    test_windows = extract_windows_multiple(test_procwave, STEP_SIZE,
                                            OVERLAP_SIZE, collapse = False)

    # construct return values
    paths = [train_paths, val_paths, test_paths]
    waveforms = [train_waveforms, val_waveforms, test_waveforms]
    processed = [train_procwave, val_procwave, test_procwave]
    wparams = [train_wparams, val_wparams, test_wparams]
    windows = [train_windows, val_windows, test_windows]

    return paths, waveforms, processed, wparams, windows









