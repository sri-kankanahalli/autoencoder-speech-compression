from __future__ import absolute_import, print_function, division
import numpy as np
import math
import theano
import theano.tensor as T
from scipy.fftpack import dct as scidct
from params import *

# generate square dct matrix
#     how to use: generate n-by-n matrix M
#                 dct(w) = M * w
#     where w must be n-by-1
#
#     identical to dct(w, norm = 'ortho') using scipack
#

def dct_mat_np(n):
    rval = np.zeros((n, n))
    col_range = np.arange(n)
    scale = np.sqrt(2.0 / n)

    for i in xrange(n):
        rval[i] = np.cos(i * (col_range * 2 + 1) / (2.0 * n) * np.pi) * scale

    rval[0] /= np.sqrt(2)

    return rval

# DCT matrix is currently precomputed at start of program into a NumPy array, then 
# shoved into a Theano tensor for later use
#     (TODO: compute at runtime. or not. please not)
dctMat = dct_mat_np(200)
th_dctMat = theano.shared(dctMat)

# quick dumb test function (comparing it to scipy's DCT)
def tst_dct():
    dct = dct_mat_np(4)
    tst = np.array([ [[1],[2],[3],[4]], [[5],[6],[7],[8]] ])

    th_dct = theano.shared(dctMat)
    th_tst = theano.shared(tstMat)

    print (th_tst.shape.eval())
    print (th_dct.shape.eval())

    result = T.tensordot(th_dct, th_tst, [[1], [1]])
    result = result.reshape((result.shape[1], result.shape[0], result.shape[2]))
    print (result.eval())

    tstMat2 = np.array([ [1, 2, 3, 4], [5, 6, 7, 8] ])
    print(scidct(tstMat2, norm = 'ortho'))

    return

# given a (symbolic Theano) array of size M x 200
#     this returns an array M x 200 x 1 where every one of the M samples has been independently
#     filtered by the DCT
def theano_dct(x):
    global th_dctMat

    # reshape x into 2D array
    reshaped_x = x.reshape((x.shape[0], x.shape[1], 1))

    result = T.tensordot(th_dctMat, reshaped_x, [[1], [1]])
    result = result.reshape((result.shape[1], result.shape[0], result.shape[2]))

    return result


'''
# theano function to generate square n-by-n DCT matrix
def dct_matrix(n):
    rval = T.zeros((n, n))
    col_range = T.arange(n)
    scale = T.sqrt(2.0 / n)

    listIn = T.arange(10000)

    components, updates = theano.scan(fn=lambda power: power,
                                      outputs_info=None,
                                      sequences=[col_range])

    calculate_polynomial = theano.function(inputs=[], outputs=components)
    print ( calculate_polynomial() )


    #print (f([0 1 2]))
    #print (results)

    i = 0
    s = T.cos(i * (col_range * 2 + 1) / (2.0 * n) * 3.14159) * scale
    rval = T.set_subtensor(rval[i], s)
    

    rval = T.set_subtensor(rval[0], rval[0] / T.sqrt(2))

    return rval
'''





