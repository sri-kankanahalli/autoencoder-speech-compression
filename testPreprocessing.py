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

from windowingFunctions import *
from params import *
from utility import *
from preprocessing import *

fileToRead = "./fiveYears.wav"
if (len(sys.argv) > 1):
    fileToRead = sys.argv[1]

[rate, data] = sciwav.read(fileToRead)
windows = extractWindows(data)

print "before preprocessing: ", windows.shape

windows = processWindows(windows)
computeMeanVariance(windows)
windows = normalizeWindows(windows)
print "after preprocessing: ", windows.shape

windows = denormalizeWindows(windows)
windows = deprocessWindows(windows)
print "after un-preprocessing: ", windows.shape

reconstruction = reconstructFromWindows(windows)
print reconstruction.shape

sciwav.write("out.wav", rate, reconstruction)

r = reconstruction[:data.shape[0]]
print "mse: ", mse(r, data)
print "avg err: ", avgErr(r, data)



