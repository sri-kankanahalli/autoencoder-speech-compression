import numpy as np
import math
from params import *

# MSE between two numpy arrays of the same size
def mse(a, b):
    return ((a - b) ** 2).mean(axis = None)
    
# average error betwene two numpy arrays of the same size
def avgErr(a, b):
    return (abs(a - b)).mean(axis = None)

