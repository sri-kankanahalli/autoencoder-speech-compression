# ==========================================================================
# functions to handle windowing of waveforms (extracting windows from a
# waveform, and reconstructing a waveform from individual windows)
# ==========================================================================

import numpy as np
import math

from consts import *

# extract overlapping windows from waveform
def extract_windows(data):
    numWindows = int(math.ceil(float(len(data)) / STEP_SIZE))

    # loop over every window
    windows = []
    for i in xrange(0, numWindows):
        # get the frame
        startOfFrame = STEP_SIZE * i
        endOfFrame = startOfFrame + WINDOW_SIZE
        frame = data[startOfFrame:endOfFrame]

        # pad frame to proper size, if there's not enough data
        if len(frame) < WINDOW_SIZE:
            frame = np.pad(frame, (0, WINDOW_SIZE - len(frame)), \
                           'constant', constant_values=[0])

        if (i == 0):
            windows = np.reshape(np.array(frame), (1, len(frame)))
        else:
            tmp = np.reshape(np.array(frame), (1, len(frame)))
            windows = np.append(windows, tmp, axis = 0)

    windows = windows.astype(np.float32)

    if (EXTRACT_MODE == 0):
        for i in xrange(0, windows.shape[0]):
            windows[i] *= WINDOWING_MULT    

    return windows

# reconstruct waveform from overlapping windows
def reconstruct_from_windows(windows):
    reconstruction = []
    lastWindow = []

    for i in xrange(0, windows.shape[0]):
        r = windows[i, :]
        
        if (i == 0):
            reconstruction = r
        else:
            overlapLastWindow = reconstruction[-OVERLAP_SIZE:]
            overlapThisWindow = r[:OVERLAP_SIZE]
            unmodifiedPart = r[OVERLAP_SIZE:]

            overlappedPart = np.copy(overlapLastWindow)
            for j in xrange(0, OVERLAP_SIZE):
                if (EXTRACT_MODE == 1):
                    thisMult = OVERLAP_FUNC[j]
                    lastMult = OVERLAP_FUNC[j + OVERLAP_SIZE]
                else:
                    thisMult = 1.0
                    lastMult = 1.0

                # use windowing function
                overlappedPart[j] = overlapThisWindow[j] * thisMult + \
                                    overlapLastWindow[j] * lastMult

            reconstruction[-OVERLAP_SIZE:] = overlappedPart
            reconstruction = np.concatenate([reconstruction, unmodifiedPart])
        
    return reconstruction

# extract windows for list of waveforms
def extract_windows_multiple(wavelist, collapse = False):
    windowlist = []
    for waveform in wavelist:
        windows = extract_windows(waveform)

        if (windowlist == []):
            windowlist = [windows]
        else:
            windowlist += [windows]

    if (collapse):
        windowlist = np.array([i for z in windowlist for i in z])

    return windowlist






