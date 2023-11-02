
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy
import h5py
import pickle
import pandas as pd
from elephant import statistics
import quantities as pq
from neo.core import SpikeTrain


def run_psth(side, file_path, file_name, output_folder, animal = 'F1702'):
    if animal == 'F1306' or animal == 'F1405':
        data = pd.read_csv(file_path + file_name, delimiter='\t')
        # recblock is in the name of the file
        block = file_name.split()[3]
        # remove the .txt
        block = block[:-4]
    else:
        data = scipy.io.loadmat(file_path + file_name)
        block = data['recBlock']
    try:
        if animal == 'F1306' or animal == 'F1405':
            freqs = data['Pitch']
            lvls = data['Atten']
            lvls = 80 - lvls
        else:
            freqs = data['currTrialDets']['Freq'][0][0].flatten()
            lvls = data['currTrialDets']['Lvl'][0][0].flatten()
            lvls = 80 - lvls
    except:
        print('no freqs')
        return
    #remove the last freqwuency
    if animal == 'F1306' or animal == 'F1405':
        fname = f'{output_folder}spikes{block}right.pkl'

    else:
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
    print('spikes shape:')
    print(spikes.shape)
    if animal == 'F1306' or animal == 'F1405':
        if side == 'left':
            #take only the first 16 channels
            spikes = spikes[:16, :]
        else:
            spikes = spikes[16:, :]
    freqs = freqs[:len(spikes[0])]
    lvls = lvls[:len(spikes[0])]

    spike_counts = {}


    sumspikes = np.zeros((len(spikes), len(spikes[0])))
    for i in range(len(spikes)):
        for i2 in range(len(spikes[i])):
            spikesintrial = spikes[i][i2]
            #filter spikes in trial to be between 0.2 and 0.3 seconds as epoch was -0.2 s before stim
            if animal == 'F1702' or animal == 'F1604' or animal == 'F1306' or animal == 'F1405':
                try:
                    spikesintrial = spikesintrial[(spikesintrial >= int(0.1*24414.0625)) & (spikesintrial <= int(0.8*24414.0625))]
                except:
                    print('error')
                    print(spikesintrial)
            else:
                try:
                   spikesintrial = spikesintrial[(spikesintrial >= int(0.2*24414.0625)) & (spikesintrial <= int(0.3*24414.0625))]
                except:
                    print('error, no spikes')
                    print(spikesintrial)
            sumspikes[i][i2] = len(spikesintrial)

    sumspikes_t_test_before = np.zeros((len(spikes), len(spikes[0])))
    sumspikes_t_test_after = np.zeros((len(spikes), len(spikes[0])))

    for i in range(len(spikes)):
        for i2 in range(len(spikes[i])):
            spikesintrial = spikes[i][i2]
            # filter spikes in trial to be between 0.1 and 0.3 seconds as epoch was -0.2 s before stim
            try:
                spikesintrial_beforestim = spikesintrial[
                    (spikesintrial >= (0.1 * 24414.0625)) & (spikesintrial < (0.2 * 24414.0625))]
                spikesintrial_afterstim = spikesintrial[
                    (spikesintrial <= (0.2 * 24414.0625)) & (spikesintrial < (0.3 * 24414.0625))]

            except:
                print('error')
                print(spikesintrial)
            try:
                sumspikes_t_test_before[i][i2] = len(spikesintrial_beforestim)
            except:
                sumspikes_t_test_before[i][i2] = 0
            try:
                sumspikes_t_test_after[i][i2] = len(spikesintrial_afterstim)
            except:
                sumspikes_t_test_after[i][i2] = 0



    #now conduct a students' t test to see if the spikes from 0.1 to 0.2s are significantly different from 0.2 to 0.3s
    #get the mean spikes from 0.1 to 0.2s but iterate over each channel
    ns_channel_list = []
    for i in range(len(sumspikes_t_test_before)):

        spikes_1 = sumspikes_t_test_before[i, :]
        spikes_2 = sumspikes_t_test_after[i, :]
        #get the number of trials
        t_statistic, p_value = scipy.stats.ttest_ind(spikes_1, spikes_2, equal_var=True)

        if p_value <= 0.05:
            # print('significant')
            #if the mean spikes from 0.1 to 0.2s are greater than the mean spikes from 0.2 to 0.3s, then the channel is on
            soundonset_channel = i
        else:
            print('not significant')
            ns_channel_list.append(i)




    if side == 'right':

        if animal == 'F1306' or animal == 'F1405':
            orderofwarpelectrodescruella_right = np.fliplr([[3,7,11,15],
                                                  [2,6,10,14],
                                                    [1,5,9,13],
                                                    [0,4,8,12]] )

        else:
            orderofwarpelectrodescruella_right = np.fliplr([[30, 31, 14, 15],
                                                   [28, 29, 12 ,13],
                                                   [26 ,27 ,10 ,11],
                                                   [24 ,25, 8, 9],
                                                   [23, 22, 7 ,6],
                                                   [21, 20, 5, 4],
                                                   [19, 18, 3, 2],
                                                   [17, 16, 1, 0]])
    else:
        if animal == 'F1306' or animal == 'F1405':
            orderofwarpelectrodescruella_right = [[3,7,11,15],
                                                  [2,6,10,14],
                                                    [1,5,9,13],
                                                    [0,4,8,12]]

        else:
            orderofwarpelectrodescruella_right = ([[30, 31, 14, 15],
                                               [28, 29, 12 ,13],
                                               [26 ,27 ,10 ,11],
                                               [24 ,25, 8, 9],
                                               [23, 22, 7 ,6],
                                               [21, 20, 5, 4],
                                               [19, 18, 3, 2],
                                               [17, 16, 1, 0]])


    if animal == 'F1306' or animal == 'F1405':
        warp_electrodes = np.array([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15,
        ])
        tdt_channels = np.array([
            1, 3, 5, 7, 2, 4, 6, 8, 10, 12,
            14, 16, 9, 11, 13, 15
        ]) - 1

    else:

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

    if animal == 'F1306' or animal == 'F1405':
        date = file_name.split()
        caldate = date[0]
    else:
        date = file_name.split('_')
        caldate = date[3]

    #combine the two arrays
    combined = np.vstack((warp_electrodes, tdt_channels))



    f32file = 0
    for i in tdt_order.flatten():

        electrode = combined[0, np.where(combined[1, :] == i)]
        # spikes = np.rot90(spikes, -1)
        if animal == 'F1306' or animal == 'F1405':
            ax = plt.subplot(4, 4, int(electrode[0][0]) + 1)
        else:
            ax = plt.subplot(8, 4, int(electrode[0][0]) + 1)
        #force colorbar to be the same for all plots

        spikes_from_electrode = spikes[i, :]
        #divide by the sample rate
#ValueError: If the input is not a spiketrain(s), it must be an MxN numpy array, each cell of which represents the number of (binned) spikes that fall in an interval - not raw spike times.

        #make a list of spike trains
        spike_trains = []
        sampling_rate = 24414.0625  # Hz

        for j in range(len(spikes_from_electrode)):#
            spikesintrial = spikes_from_electrode[j]

            if animal == 'F1702' or animal == 'F1604' or animal == 'F1306' or animal == 'F1405':
                try:
                    spikesintrial = spikesintrial[
                        (spikesintrial >= int(0.1 * 24414.0625)) & (spikesintrial <= int(0.8 * 24414.0625))]
                    spikesintrial = spikesintrial/sampling_rate

                except:
                    print('error')
                    print(spikesintrial)
            else:
                try:
                    spikesintrial = spikesintrial[
                        (spikesintrial >= int(0.2 * 24414.0625)) & (spikesintrial <= int(0.3 * 24414.0625))]
                    spikesintrial = spikesintrial/sampling_rate
                except:
                    print('error, no spikes')
                    print(spikesintrial)
                    continue

            # spike_times_with_units = spikesintrial * pq.s
            #
            #
            # spike_train = SpikeTrain(spike_times_with_units, t_start=0.2 * pq.s, t_stop=0.3* pq.s)
            spike_trains.append(spikesintrial)
        #concatenate the spike trains
        conc_spks = np.concatenate(spike_trains)


        trials = [np.ones(len(spikes[i, j])) * j for j in range(len(spikes[0, :]))]
        psth = np.histogram(conc_spks, bins=100)
        ax.plot(psth[0])
        ax.set_title(f'Channel {i + 1}')

        if i in ns_channel_list:
            plt.axvspan(0, spikes.shape[1] - 1, facecolor='grey', alpha=0.5)


        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.title(f'Channel {i + 1}', fontsize=10)

        #have one giant colorbar
        # if i == 31:
        #     plt.colorbar(label='Spikes per presentation')

    #increase the space between the plots
    #increase the size of the figure
    if animal == 'F1306' or animal == 'F1405':
        plt.gcf().set_size_inches(10, 10)
    else:
        plt.gcf().set_size_inches(10, 15)
    plt.subplots_adjust(wspace=0.6, hspace=0.7)
    #save figure in output folder
    #extract the date from the file name
    if animal == 'F1306' or animal == 'F1405':
        plt.suptitle(f'PSTH for {block}, {side} side {animal}, {caldate}', fontsize=16)

        plt.savefig(f'{output_folder}PSTH_for_{block}_{caldate}{side}_side_'+animal+'.pdf', dpi = 300, bbox_inches='tight')
        plt.savefig(f'{output_folder}PSTH_for_{block}_{caldate}{side}_side_' + animal + '.png', dpi=300,
                    bbox_inches='tight')
    else:
        plt.suptitle(f'PSTH for {block[0]}, {side} side {animal}, {caldate}', fontsize=16)

        plt.savefig(f'{output_folder}PSTH_for_{block[0]}_{caldate}{side}_side_' + animal + '.pdf', dpi=300,
                    bbox_inches='tight')
        plt.savefig(f'{output_folder}PSTH_for_{block[0]}_{caldate}{side}_side_' + animal + '.png', dpi=300,
                    bbox_inches='tight')


    plt.show()

    return block



