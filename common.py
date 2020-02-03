########################################################################
# import python-library
########################################################################
# default
import glob
import argparse
import sys

# additional
import numpy
import librosa
import librosa.core
import librosa.feature
import yaml

# from import
#from keras.models import Model
#from keras.layers import Input, Dense
########################################################################

########################################################################
# version
########################################################################
__versions__ = "1.0.0"
########################################################################

########################################################################
# argparse
########################################################################
def command_line_chk():
    parser = argparse.ArgumentParser(description='Without option argument, it will not run properly.')

    parser.add_argument('-v', '--version', action='store_true', help="show application version")
    parser.add_argument('-e', '--eval', action='store_true', help="run mode Evaluation")
    parser.add_argument('-d', '--dev', action='store_true', help="run mode Development")

    args = parser.parse_args()
    if args.version:
        print("===============================")
        print("DCASE 2020 task 2 baseline\nversion {}".format(__versions__))
        print("===============================\n")
    if args.eval ^ args.dev:
        if args.dev:
            flag = True
        else:
            flag = False
    else:
        flag = None
        print("incorrect argument")
        print("please set option argument '-dev' or '-eval'")
    return flag
########################################################################


########################################################################
# load parameter.yaml
########################################################################
def yaml_load():
    with open("baseline.yaml") as stream:
        param = yaml.load(stream)
    return param


"""
dev_directory : ./dev_data
eval_directory : ./eval_data
model_directory: ./model
result_directory: ./result
result_file: result.csv


max_fpr : 0.1

feature:
  n_mels: 64
  frames : 5
  n_fft: 1024
  hop_length: 512
  power: 2.0


fit:
  compile:
    optimizer : adam
    loss : mean_squared_error
  epochs : 100
  batch_size : 512
  shuffle : True
  validation_split : 0.1
  verbose : 1

"""

########################################################################


########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "baseline.log".
"""
import logging

logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


########################################################################


########################################################################
# file I/O
########################################################################
# wav file Input
def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for monoral data

    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error(f'{"file_broken or not exists!! : {}".format(wav_name)}')


########################################################################


########################################################################
# feature extractor
########################################################################
def file_to_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, fearture_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa
    y, sr = file_load(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vector_array_size < 1:
        return numpy.empty((0, dims), float)

    # 06 generate feature vectors by concatenating multiframes
    vector_array = numpy.zeros((vector_array_size, dims), float)
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

    return vector_array


# load dataset
def select_dirs(param, mode):
    """
    param : dict
        baseline.yaml data

    return :
        if active type the development :
            dirs :  list [ str ]
                load base directory list of dev_data
        if active type the evaluation :
            dirs : list [ str ]
                load base directory list of eval_data
    """
    if mode:
        logger.info("load_directory <- development")
        dirs = sorted(glob.glob("{base}/*".format(base=param["dev_directory"])))
    else:
        logger.info("load_directory <- evaluation")
        dirs = sorted(glob.glob("{base}/*".format(base=param["eval_directory"])))
    return dirs

########################################################################

