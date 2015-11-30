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

[rate, data] = sciwav.read("fiveYears.wav")
print data.shape

windows = extractWindows(data)
print windows.shape

reconstruction = reconstructFromWindows(windows)
print reconstruction.shape

sciwav.write("out.wav", rate, reconstruction)

r = reconstruction[:data.shape[0]]
print "mse: ", mse(r, data)
print "avg err: ", avgErr(r, data)


