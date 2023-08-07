
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
        bounds, bf, Th, data, spikes, levels = FRAbounds(FRAinput, f32file)
        #plot the spikes in a heatmap in a 4 x8 grid
        plt.subplot(4, 8, i + 1)
        #force colorbar to be the same for all plots
        plt.imshow(spikes, origin='lower', aspect='auto', cmap='hot')
        # plt.clim(0, 10)
        if i == 24:
            plt.xticks(np.linspace(0, spikes.shape[1] - 1, num=6),
                       np.round(np.exp(np.linspace(np.log(min(freqs)), np.log(max(freqs)), num=6)) / 1000, 2), fontsize=8)
            plt.yticks(np.linspace(0, spikes.shape[0] - 1, num=6),
                       np.round(np.linspace(min(levels), max(levels), num=6), 2), fontsize=8)
            plt.xlabel('Freq (kHz)', fontsize=10)
            plt.ylabel('Level (dB)', fontsize=10)

        plt.colorbar()
        plt.title(f'Channel {i + 1}', fontsize=10)

        #have one giant colorbar
        # if i == 31:
        #     plt.colorbar(label='Spikes per presentation')

    #increase the space between the plots
    #increase the size of the figure
    plt.gcf().set_size_inches(15, 10)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    plt.show()

    return block
    #


