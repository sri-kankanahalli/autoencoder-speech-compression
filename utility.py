import numpy as np
import math
import os

# MSE between two numpy arrays of the same size
def mse(a, b):
    return ((a - b) ** 2).mean(axis = None)
    
# average error betwene two numpy arrays of the same size
def avgErr(a, b):
    return (abs(a - b)).mean(axis = None)

# get list of files in directory
def filesInDir(dirName):
    fileList = next(os.walk(dirName))[2]
    for i in xrange(0, len(fileList)):
        fileList[i] = dirName + fileList[i]
    return fileList
