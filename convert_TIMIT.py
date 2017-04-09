# ==========================================================================
# utility script to convert all TIMIT .wavs from "NIST Sphere" format into
# an actually usable .wav format
#     (mono, 16KHz, 32-bit)
#
# modified from:
#     https://fieldarchives.wordpress.com/2014/02/18/converting-the-wav-files/
# this script requires the 'sox' utility
# ==========================================================================

from scipy.io import wavfile
import subprocess
import glob
import os
from sys import stdout

from consts import *

# Lists all the wav files
train_files_list = glob.glob(TIMIT_DIR + '/TIMIT/TRAIN/*/*/*.WAV')
test_files_list = glob.glob(TIMIT_DIR + '/TIMIT/TEST/*/*/*.WAV')

# assert that our training/testing splits are correct
assert (len(train_files_list) == 4620)
assert (len(test_files_list) == 1680)

# combine training/testing lists into one giant list
wav_files_list = train_files_list + test_files_list
 
# create temporary names for the wav files to be converted (they will be renamed later on)
wav_prime=[]
for f in wav_files_list:
    fileName, fileExtension = os.path.splitext(f)
    fileName += 'b'
    wav_prime.append(fileName+fileExtension)

# command strings
convert_cmd = "sox {0} -t wav {1}"
mv_cmd = "mv {0} {1}"
 
# 1. convert the wav_files first
# 2. remove it
# 3. rename the new file created by sox to its original name
for i, f in enumerate(wav_files_list):
    subprocess.call(convert_cmd.format(f, wav_prime[i]), shell=True)
    os.remove(f)
    subprocess.call(mv_cmd.format(wav_prime[i],f), shell=True)

    print (str(i) + ": " + f + "\r"),



