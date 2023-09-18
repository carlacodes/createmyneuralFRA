
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from helpers.FRAbounds2 import FRAbounds
import scipy
import h5py
import pickle
import pandas as pd




def run_fra(side, file_path, file_name, output_folder, animal = 'F1702'):
    if animal == 'F1306':
        data = pd.read_csv(file_path + file_name, delimiter='\t')
        # recblock is in the name of the file
        block = file_name.split()[3]
        # remove the .txt
        block = block[:-4]
    else:
        data = scipy.io.loadmat(file_path + file_name)
        block = data['recBlock']
    try:
        if animal == 'F1306':
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
            #filter spikes in trial to be between 0.2 and 0.3 seconds as epoch was -0.2 s before stim
            if animal == 'F1702' or 'F1604':
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
        # #calculate two-sided t test statistic
        #
        # #calculate the mean of the spikes from 0.1 to 0.2s
        # mean_spikes_1 = np.mean(spikes_1)
        # #calculate the mean of the spikes from 0.2 to 0.3s
        # mean_spikes_2 = np.mean(spikes_2)
        # #calculate the standard deviation of the spikes from 0.1 to 0.2s
        # std_spikes_1 = np.std(spikes_1)
        # #calculate the standard deviation of the spikes from 0.2 to 0.3s
        # std_spikes_2 = np.std(spikes_2)
        # #calculate the number of trials
        # n_1 = len(spikes_1)
        # n_2 = len(spikes_2)
        # #calculate the standard error of the mean
        # sem_1 = std_spikes_1/np.sqrt(n_1)
        # sem_2 = std_spikes_2/np.sqrt(n_2)
        # #calculate the standard error of the difference between the means
        # sed = np.sqrt(sem_1**2.0 + sem_2**2.0)
        # #calculate the t statistic
        # t_statistic = (mean_spikes_1 - mean_spikes_2) / sed
        # #compare the t statistic to the critical t value
        # df = n_1 + n_2 - 2
        # alpha = 0.05
        # #calculate the critical t value
        # cv = scipy.stats.t.ppf(1.0 - alpha, df)
        # #calculate the p value
        # p = (1.0 - scipy.stats.t.cdf(abs(t_statistic), df)) * 2.0
        # #print the results
        # print('t=%.3f, df=%d, cv=%.3f, p=%.3f' % (t_statistic, df, cv, p))
        if p_value <= 0.05:
            # print('significant')
            #if the mean spikes from 0.1 to 0.2s are greater than the mean spikes from 0.2 to 0.3s, then the channel is on
            soundonset_channel = i
        else:
            print('not significant')
            ns_channel_list.append(i)


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

    if side == 'right':
        orderofwarpelectrodescruella_right = np.fliplr([[30, 31, 14, 15],
                                               [28, 29, 12 ,13],
                                               [26 ,27 ,10 ,11],
                                               [24 ,25, 8, 9],
                                               [23, 22, 7 ,6],
                                               [21, 20, 5, 4],
                                               [19, 18, 3, 2],
                                               [17, 16, 1, 0]])
    else:
        orderofwarpelectrodescruella_right = ([[30, 31, 14, 15],
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
        try:
            FRAinput = np.empty((len( spike_counts[i]), 3))

            FRAinput[:, 0] = spike_counts[i]

            FRAinput[:, 1] = freqs[:len(spike_counts[i])]
            FRAinput[:, 2] = lvls[:len(spike_counts[i])]
        except:
            FRAinput = np.empty((len(freqs), 3))

            spikecountsfortrial = spike_counts[i]
            FRAinput[:, 0] = spikecountsfortrial[:len(freqs)]
            FRAinput[:, 1] = freqs[:]
            FRAinput[:, 2] = lvls[:]

        #transpose the matrix
        # FRAinput = FRAinput.T
        bounds, bf, Th, data, spikes, levels = FRAbounds(FRAinput, f32file)
        #plot the spikes in a heatmap in a 4 x8 grid
        #find where i is in the combined mat
        #find the corresponding tdt channel
        #find the corresponding electrode
        #flip spikes up down
        # spikes = np.flipud(spikes)
        #transpose spikes
        # spikes = spikes.T

        electrode = combined[0, np.where(combined[1, :] == i)]
        # spikes = np.rot90(spikes, -1)

        ax = plt.subplot(8, 4, int(electrode[0][0]) + 1)
        #force colorbar to be the same for all plots
        ax.imshow(spikes, origin='lower', aspect='auto',cmap='hot')
        #plot the bounds as white lines
        #plot the bounds as contiguous lines
        x_bounds = np.arange(len(bounds))

        # Plot the bounds as contiguous lines
        ax.plot(x_bounds, bounds, color='white', linewidth=2)        #remove that white space around the plot
        #plot the bounds on the plot without shrinking the image


        # for k, bound in enumerate(bounds):
        #     if bound < spikes.shape[0]:
        #         plt.plot(k, bound, 'w.', markersize=10)
                # plt.plot([k - 0.5, k + 0.5], [bound, bound], color='white', linewidth=2)

        # plt.clim(0, 10)
        #if i is in ns_channel list then plot a grey box over the imshow plot

        if i in ns_channel_list:
            plt.axvspan(0, spikes.shape[1] - 1, facecolor='grey', alpha=0.5)
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

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(ax.imshow(spikes, origin='lower', aspect='auto', cmap='hot'), cax=cax)
        plt.title(f'Channel {i + 1}', fontsize=10)

        #have one giant colorbar
        # if i == 31:
        #     plt.colorbar(label='Spikes per presentation')

    #increase the space between the plots
    #increase the size of the figure

    plt.gcf().set_size_inches(10, 15)
    plt.subplots_adjust(wspace=0.6, hspace=0.7)
    plt.suptitle(f'FRA for {block[0]}, {side} side {animal}, {caldate}', fontsize=16)
    #save figure in output folder
    #extract the date from the file name

    plt.savefig(f'{output_folder}FRA_for_{block[0]}_{caldate}{side}_side_'+animal+'.pdf', dpi = 300, bbox_inches='tight')
    plt.savefig(f'{output_folder}FRA_for_{block[0]}_{caldate}{side}_side_' + animal + '.png', dpi=300,
                bbox_inches='tight')


    plt.show()

    return block
    #


