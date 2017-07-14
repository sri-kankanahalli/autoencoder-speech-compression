# ==========================================================================
# functions to handle windowing of waveforms (extracting windows from a
# waveform, and reconstructing a waveform from individual windows)
# ==========================================================================

import numpy as np
import math

# extract overlapping windows from waveform
def extract_windows(data, stepSize, overlapSize):
    numWindows = int(math.ceil(float(len(data)) / stepSize))
    windowSize = stepSize + overlapSize

    # loop over every window
    windows = []
    for i in xrange(0, numWindows):
        # get the frame
        startOfFrame = stepSize * i
        endOfFrame = startOfFrame + windowSize
        frame = data[startOfFrame:endOfFrame]

        # pad frame to proper size, if there's not enough data
        if len(frame) < windowSize:
            frame = np.pad(frame, (0, windowSize - len(frame)), \
                           'constant', constant_values=[0])

        if (i == 0):
            windows = np.reshape(np.array(frame), (1, len(frame)))
        else:
            tmp = np.reshape(np.array(frame), (1, len(frame)))
            windows = np.append(windows, tmp, axis = 0)

    windows = windows.astype(np.float32)
    
    return windows

# reconstruct waveform from overlapping windows
def reconstruct_from_windows(windows, overlapSize, overlapFunc):
    reconstruction = []
    lastWindow = []

    for i in xrange(0, windows.shape[0]):
        r = windows[i, :]
        
        if (i == 0):
            reconstruction = r
        else:
            overlapLastWindow = reconstruction[-overlapSize:]
            overlapThisWindow = r[:overlapSize]
            unmodifiedPart = r[overlapSize:]

            overlappedPart = np.copy(overlapLastWindow)
            for j in xrange(0, overlapSize):
                thisMult = overlapFunc[j]
                lastMult = overlapFunc[j + overlapSize]

                # use windowing function
                overlappedPart[j] = overlapThisWindow[j] * thisMult + \
                                    overlapLastWindow[j] * lastMult

            reconstruction[-overlapSize:] = overlappedPart
            reconstruction = np.concatenate([reconstruction, unmodifiedPart])
        
    return reconstruction

# extract windows for list of waveforms
def extract_windows_multiple(wavelist, stepSize, overlapSize,
                             collapse = False):
    windowlist = []
    for waveform in wavelist:
        windows = extract_windows(waveform, stepSize, overlapSize)

        if (windowlist == []):
            windowlist = [windows]
        else:
            windowlist += [windows]

    if (collapse):
        windowlist = np.array([i for z in windowlist for i in z])

    return windowlist






