
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from helpers.FRAbounds2 import FRAbounds
import scipy
import h5py
import pickle




def run_fra(side, file_path, file_name, output_folder):
    data = scipy.io.loadmat(file_path + file_name)
    block = data['recBlock']

    freqs = data['currTrialDets']['Freq'][0][0].flatten()
    #remove the last freqwuency
    lvls = data['currTrialDets']['Lvl'][0][0].flatten()
    lvls = 80 - lvls
    fname = f'{output_folder}spikes{block[0]}{side}.pkl'
    #read the pkl file
    try:
        with open(fname, 'rb') as f:
            try:
                spikes_data = pickle.load(f)
            except EOFError:
                print('Empty file!')
                return
    except FileNotFoundError:
        print('File not found!')
        return
    # spikes_data = h5py.File(fname, 'r')

    spikes = spikes_data
    freqs = freqs[:len(spikes[0])]
    lvls = lvls[:len(spikes[0])]

    spike_counts = {}


    sumspikes = np.zeros((len(spikes), len(spikes[0])))
    for i in range(len(spikes)):
        for i2 in range(len(spikes[i])):
            spikesintrial = spikes[i][i2]
            sumspikes[i][i2] = len(spikesintrial)



    # avgspikes = avgspikes / 1.2
    # / 1.2
    #
    # f32file = 0
    # top_row = avgspikes[0, :]
    # #make top_row, freqs and levels a column in an array
    # fra_input = np.empty((len(top_row), 3))
    # fra_input[:, 0] = top_row
    # fra_input[:, 1] = freqs[:776]
    # fra_input[:, 2] = lvls[:776]
    #
    f32file = 0
    for i in range(0, 32):
        spike_counts[i] = sumspikes[i, :]
        FRAinput = np.empty((len(spike_counts[i]), 3))
        FRAinput[:, 0] = spike_counts[i]
        FRAinput[:, 1] = freqs[:len(spike_counts[i])]
        FRAinput[:, 2] = lvls[:len(spike_counts[i])]
        #transpose the matrix
        # FRAinput = FRAinput.T
        bounds, bf, Th, data, spikes = FRAbounds(FRAinput, f32file)
    return block
    #


