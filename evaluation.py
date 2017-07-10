# ==========================================================================
# functions to evaluate a trained model
# ==========================================================================
from nn_util import *
from pesq import *
from consts import *
from load_data import *

from keras.models import *
import numpy as np
import scipy.io.wavfile as sciwav

# MSE between two numpy arrays of the same size
def np_mse(a, b):
    return ((a - b) ** 2).mean(axis = None)
    
# average error betwene two numpy arrays of the same size
def np_avgErr(a, b):
    return (abs(a - b)).mean(axis = None)

# return desired and reconstructed waveforms, from speech windows
def run_model_on_windows(windows, wparams, autoencoder, argmax = False):
    # first, get desired reconstruction
    desired = reconstruct_from_windows(windows, OVERLAP_SIZE)
    desired = unpreprocess_waveform(desired, wparams)
    desired = np.clip(desired, -32767, 32767)
    
    # then, run NN on windows to get our model's reconstruction
    enc = autoencoder.layers[1]
    
    embed = enc.predict(windows, batch_size = 128, verbose = 0)
    if (type(embed) is list or type(embed) is tuple):
        embed = embed[0]

    if (argmax):
        for wnd in xrange(0, embed.shape[0]):
            max_idxs = np.argmax(embed[wnd], axis = -1)
            embed[wnd] = np.eye(NBINS)[max_idxs]
    
    dec = autoencoder.layers[2]
    autoencOutput = dec.predict(embed, batch_size = 128, verbose = 0)
    recons = reconstruct_from_windows(autoencOutput, OVERLAP_SIZE)
    recons = unpreprocess_waveform(recons, wparams)
    recons = np.clip(recons, -32767, 32767)
    
    return desired, recons

# return desired and reconstructed waveforms, from .wav filename
def run_model_on_wav(wave_filename, autoencoder, argmax = False):
    [rate, data] = sciwav.read(wave_filename)
    data = data.astype(np.float32)
    processed_wave, wparams = preprocess_waveform(data)
    windows = extract_windows(processed_wave, STEP_SIZE, OVERLAP_SIZE,
                              WINDOWING_MULT)
    
    desired, recons = run_model_on_windows(windows, wparams, autoencoder, argmax)
    
    return desired, recons

# return evaluation metrics, given desired and reconstructed waveforms
def evaluation_metrics(desired, recons):
    pesq = run_pesq_waveforms(desired, recons)
    
    # return some metrics, as well as the two waveforms
    metrics = [
        np_mse(recons, desired),
        np_avgErr(recons, desired),
        pesq
    ]
    
    return metrics

# test model on a set of speech windows (which should originally have been extracted, in
# order, from some speech waveform)
def test_model_on_windows(windows, wparams, autoencoder, argmax = False):
    # compute PESQ between desired and reconstructed waveforms
    desired, recons = run_model_on_windows(windows, wparams, autoencoder, argmax)
    return evaluation_metrics(desired, recons), desired, recons

# test model given the filename for a .wav file
def test_model_on_wav(wave_filename, prefix, autoencoder,
                      lead = "", save_recons = True, verbose = True,
                      argmax = False):
    desired, recons = run_model_on_wav(wave_filename, autoencoder, argmax)
    metrics = evaluation_metrics(desired, recons)
    
    if (save_recons):
        outFilename = prefix + "_output.wav"
        sciwav.write(outFilename, SAMPLE_RATE, recons.astype(np.int16))
    
    if (verbose):
        print lead + "MSE:        ", metrics[0]
        print lead + "Avg err:    ", metrics[1]
        print lead + "PESQ:       ", metrics[2]
    
    return metrics





