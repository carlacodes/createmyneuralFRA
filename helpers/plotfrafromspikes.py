
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
    orderofwarpelectrodescruella_right = np.fliplr([[30, 31, 14, 15],
                                           [28, 29, 12 ,13],
                                           [26 ,27 ,10 ,11],
                                           [24 ,25, 8, 9],
                                           [23, 22, 7 ,6],
                                           [21, 20, 5, 4],
                                           [19, 18, 3, 2],
                                           [17, 16, 1, 0]])


    warp_electrodes = np.array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    ])
    tdt_channels = np.array([
        1, 3, 5, 7, 2, 4, 6, 8, 10, 12,
        14, 16, 9, 11, 13, 15, 17, 19, 21, 23,
        18, 20, 22, 24, 26, 28, 30, 32, 25, 27,
        29, 31
    ])-1
    tdt_order = tdt_channels[warp_electrodes[orderofwarpelectrodescruella_right]]

    print(tdt_order)
    date = file_name.split('_')
    caldate = date[3]

    #combine the two arrays
    combined = np.vstack((warp_electrodes, tdt_channels))



    f32file = 0
    for i in tdt_order.flatten():
        spike_counts[i] = sumspikes[i, :]
        FRAinput = np.empty((len(spike_counts[i]), 3))
        FRAinput[:, 0] = spike_counts[i]
        FRAinput[:, 1] = freqs[:len(spike_counts[i])]
        FRAinput[:, 2] = lvls[:len(spike_counts[i])]
        #transpose the matrix
        # FRAinput = FRAinput.T
        bounds, bf, Th, data, spikes, levels = FRAbounds(FRAinput, f32file)
        #plot the spikes in a heatmap in a 4 x8 grid
        #find where i is in the combined mat
        #find the corresponding tdt channel
        #find the corresponding electrode
        electrode = combined[0, np.where(combined[1, :] == i)]
        plt.subplot(8, 4, int(electrode[0][0]) + 1)
        #force colorbar to be the same for all plots
        plt.imshow(spikes, origin='lower', aspect='auto', cmap='hot')
        # plt.clim(0, 10)
        if i == 24:
            plt.xticks(np.linspace(0, spikes.shape[1] - 1, num=6),
                       np.round(np.exp(np.linspace(np.log(min(freqs)), np.log(max(freqs)), num=6)) / 1000, 2), fontsize=8, rotation = 45)
            plt.yticks(np.linspace(0, spikes.shape[0] - 1, num=6),
                       np.round(np.linspace(min(levels), max(levels), num=6), 2), fontsize=8)
            plt.xlabel('Freq (kHz)', fontsize=10)
            plt.ylabel('Level (dB)', fontsize=10)
        else:
            plt.xticks(np.linspace(0, spikes.shape[1] - 1, num=6),
                       np.round(np.exp(np.linspace(np.log(min(freqs)), np.log(max(freqs)), num=6)) / 1000, 2),
                       fontsize=8, rotation=45)
            plt.yticks(np.linspace(0, spikes.shape[0] - 1, num=6),
                       np.round(np.linspace(min(levels), max(levels), num=6), 2), fontsize=8)

        plt.colorbar()
        plt.title(f'Channel {i + 1}', fontsize=10)

        #have one giant colorbar
        # if i == 31:
        #     plt.colorbar(label='Spikes per presentation')

    #increase the space between the plots
    #increase the size of the figure

    plt.gcf().set_size_inches(10, 15)
    plt.subplots_adjust(wspace=0.6, hspace=0.7)
    plt.suptitle(f'FRA for {block[0]}, {side} side F1815, {caldate}', fontsize=16)
    #save figure in output folder
    #extract the date from the file name

    plt.savefig(f'{output_folder}FRA_for_{block[0]}_{caldate}{side}_side_F1815.pdf', dpi = 300, bbox_inches='tight')
    plt.show()

    return block
    #


