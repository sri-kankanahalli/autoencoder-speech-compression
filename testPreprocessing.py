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
from mfcc import *
from preprocessing import *

[rate, data] = sciwav.read("sp01.wav")
windows = extractWindows(data)

transformed = transformWindows(windows)
computeMeanVariance(transformed, windows)

print "before preprocessing: ", windows.shape

windows = preprocessOrigWindows(windows)
windows = normalizeOrigWindows(windows)
print "after preprocessing: ", windows.shape

windows = denormalizeOrigWindows(windows)
windows = unpreprocessOrigWindows(windows)
print "after un-preprocessing: ", windows.shape

reconstruction = reconstructFromWindows(windows)
print reconstruction.shape

sciwav.write("out.wav", rate, reconstruction)

r = reconstruction[:data.shape[0]]
print "mse: ", mse(r, data)
print "avg err: ", avgErr(r, data)
