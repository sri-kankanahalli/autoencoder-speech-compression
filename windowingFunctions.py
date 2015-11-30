import numpy as np
import math
from params import *

# extract overlapping windows from waveform
def extractWindows(data):
    numWindows = int(math.ceil(float(len(data)) / STEP_SIZE))

    # loop over every window
    windows = []
    for i in xrange(0, numWindows):
        # get the frame
        startOfFrame = STEP_SIZE * i
        endOfFrame = startOfFrame + WINDOW_SIZE
        frame = data[startOfFrame : endOfFrame]

        # pad frame to proper size, if there's not enough data
        #     (this should only happen for the last frame in a file)
        if len(frame) < WINDOW_SIZE:
            frame = np.pad(frame, (0, WINDOW_SIZE - len(frame)), \
                           'constant', constant_values=[0])

        if (i == 0):
            windows = np.reshape(np.array(frame), (1, len(frame)))
        else:
            tmp = np.reshape(np.array(frame), (1, len(frame)))
            windows = np.append(windows, tmp, axis = 0)

    windows = windows.astype(np.float32)
    
    return windows
    
# reconstruct waveform from overlapping windows
def reconstructFromWindows(windows):
    reconstruction = []
    lastWindow = []
    for i in xrange(0, windows.shape[0]):
        r = windows[i, :]
        
        if (i == 0):
            reconstruction = r
        else:
            currLastWindow = reconstruction[-STEP_SIZE:]
            thisLastWindow = r[:STEP_SIZE]
            nextWindow = r[STEP_SIZE:]
            
            # overlap redundant parts of window. here's a visual
            #      window 1: A   B1
            #      window 2:     B2   C
            #      merged:   A   B~   C
            firstHalf = np.copy(currLastWindow)
            for j in xrange(0, STEP_SIZE):
                # use windowing function
                thisMult = WINDOW_FUNC[j]
                lastMult = WINDOW_FUNC[j + STEP_SIZE]

                #thisMult = float(j) / STEP_SIZE
                #lastMult = 1 - thisMult
                
                firstHalf[j] = thisLastWindow[j] * thisMult + \
                               currLastWindow[j] * lastMult
            
            reconstruction[-STEP_SIZE:] = firstHalf
            reconstruction = np.concatenate([reconstruction, nextWindow])
        
    reconstruction =  reconstruction.astype(np.int16)
    return reconstruction
    
    