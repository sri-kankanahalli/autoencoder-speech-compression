from __future__ import absolute_import, print_function, division
import numpy as np
import math
import theano
import theano.tensor as T
from scipy.fftpack import dct as scidct
from params import *

# we precompute all filters at the start of the program into NumPy arrays, shoving
# them into Theano tensors for later use. so we need to know the size of all
# filters at program start
#     (TODO: dynamically generate DCT/DFT at runtime. or not. please not)
FILTER_SIZE = 320



# ====================================================================
#  DCT (Discrete Cosine Transform)
# ====================================================================

# generate square dct matrix
#     how to use: generate n-by-n matrix M. then, if you have a signal w, then:
#                 dct(w) = M * w
#     where w must be n-by-1
#
#     backed by scipy
def generate_dct_mat(n):
    return (scidct(np.eye(n), norm = 'ortho'))

# DCT matrix is precomputed at start of program
dctMat = generate_dct_mat(FILTER_SIZE)
th_dctMat = theano.shared(dctMat)

# quick dumb test function (comparing Theano matrix mults to scipy's direct DCT)
def tst_dct():
    dct = generate_dct_mat(4)
    tst = np.array([ [1, 2, 3, 4], [5, 6, 7, 8] ])

    th_dct = theano.shared(dct)
    th_tst = theano.shared(tst)

    th_tst = th_tst.reshape((1, th_tst.shape[0], th_tst.shape[1]))

    print (th_tst.shape.eval())
    print (th_dct.shape.eval())

    result = T.tensordot(th_dct, th_tst, [[0], [2]])
    result = result.reshape((result.shape[0], result.shape[2])).T
    print (result.eval())
    print (result.shape.eval())

    tst2 = np.array([ [1, 2, 3, 4], [5, 6, 7, 8] ])
    result2 = (scidct(tst2, norm = 'ortho'))
    print (result2)
    print (result2.shape)

    return

# given a (symbolic Theano) array of size M x 200
#     this returns an array M x 200 where every one of the M samples has been independently
#     filtered by the DCT
def theano_dct(x):
    global th_dctMat

    # reshape x into 2D array, and perform appropriate matrix operation
    reshaped_x = x.reshape((1, x.shape[0], x.shape[1]))

    result = T.tensordot(th_dctMat, reshaped_x, [[0], [2]])
    result = result.reshape((result.shape[0], result.shape[2])).T

    return result



# ====================================================================
#  DFT (Discrete Fourier Transform)
# ====================================================================

# generate square dft matrix (similar to how we generate the DFT one)
#     note that this matrix will have real and imaginary components
def generate_dft_mat(n):
    return (np.fft.fft(np.eye(n)))

# we compute both the real and imaginary part of the FFT separately, at program start
dftMat = generate_dft_mat(FILTER_SIZE)

th_dftMat_imag = theano.shared(np.imag(dftMat))
th_dftMat_real = theano.shared(np.real(dftMat))

# quick dumb test function (comparing Theano matrix mults to numpy's direct FFT)
def tst_dft():
    dft = generate_dft_mat(4)
    tst = np.array([ [1, 2, 3, 4], [5, 6, 7, 8] ])

    th_dft = theano.shared(dft)
    th_tst = theano.shared(tst)

    th_tst = th_tst.reshape((1, th_tst.shape[0], th_tst.shape[1]))

    print (th_tst.shape.eval())
    print (th_dft.shape.eval())

    result = T.tensordot(th_dft, th_tst, [[0], [2]])
    result = result.reshape((result.shape[0], result.shape[2])).T
    print (result.eval())
    print (result.shape.eval())

    tst2 = np.array([ [1, 2, 3, 4], [5, 6, 7, 8] ])
    result2 = (np.fft.fft(tst2))
    print (result2)
    print (result2.shape)

    # compute magnitude
    print (abs(result2))

    th_dft_imag = theano.shared(np.imag(dft))
    imag = T.tensordot(th_dft_imag, th_tst, [[0], [2]])
    imag = imag.reshape((imag.shape[0], imag.shape[2])).T
    print (imag.eval())

    th_dft_real = theano.shared(np.real(dft))
    real = T.tensordot(th_dft_real, th_tst, [[0], [2]])
    real = real.reshape((real.shape[0], real.shape[2])).T
    print (real.eval())

    both = T.concatenate([imag, real], axis=1)
    print (both.shape.eval())

    my_mag = T.sqrt(T.sqr(real) + T.sqr(imag))
    print (my_mag.eval())

    return

# given a (symbolic Theano) array of size M x 200
#     this returns an array M x 200 where every one of the M samples has been replaced by
#     its DFT magnitude
def theano_dft_mag(x):
    global th_dftMat_imag
    global th_dftMat_real

    reshaped_x = x.reshape((1, x.shape[0], x.shape[1]))

    imag = T.tensordot(th_dftMat_imag, reshaped_x, [[0], [2]])
    imag = imag.reshape((imag.shape[0], imag.shape[2])).T

    real = T.tensordot(th_dftMat_real, reshaped_x, [[0], [2]])
    real = real.reshape((real.shape[0], real.shape[2])).T

    result = T.sqrt(T.sqr(real) + T.sqr(imag))

    return result

# given a (symbolic Theano) array of size M x 200
#     this returns an array M x 400 where coefficients are alternating real and imaginary
#     FFT coeffs
def theano_dft(x):
    global th_dftMat_imag
    global th_dftMat_real

    reshaped_x = x.reshape((1, x.shape[0], x.shape[1]))

    imag = T.tensordot(th_dftMat_imag, reshaped_x, [[0], [2]])
    imag = imag.reshape((imag.shape[0], imag.shape[2])).T

    real = T.tensordot(th_dftMat_real, reshaped_x, [[0], [2]])
    real = real.reshape((real.shape[0], real.shape[2])).T

    result = T.concatenate([imag, real], axis=1)

    return result

#tst_dct()
tst_dft()
















