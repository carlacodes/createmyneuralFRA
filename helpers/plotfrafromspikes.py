
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from helpers.FRAbounds import genFRAbounds
import scipy
import h5py
import pickle




def run_fra(side, file_path, file_name, output_folder):
    data = scipy.io.loadmat(file_path + file_name)
    block = data['recBlock']

    freqs = data['currTrialDets']['Freq'][0][0].flatten()
    lvls = data['currTrialDets']['Lvl'][0][0].flatten()
    lvls = 80 - lvls
    fname = f'{output_folder}spikes{block[0]}{side}.pkl'
    #read the pkl file
    with open(fname, 'rb') as f:
        spikes_data = pickle.load(f)
    # spikes_data = h5py.File(fname, 'r')

    spikes = spikes_data




    avgspikes = np.zeros((len(spikes), len(spikes[0])))
    for i in range(len(spikes)):
        for i2 in range(len(spikes[i])):
            spikesintrial = spikes[i][i2]
            avgspikes[i][i2] = len(spikesintrial)  # / 1.2

    f32file = 0
    top_row = avgspikes[0, :]
    #make top_row, freqs and levels a column in an array
    fra_input = np.empty((len(top_row), 3))
    fra_input[:, 0] = top_row
    fra_input[:, 1] = freqs[:776]
    fra_input[:, 2] = lvls[:776]

    bounds, bf, Th, data, spikes = genFRAbounds(fra_input, f32file)

    # Plot the results
    os.makedirs('D:/Data/PSTHresults/Cruella/Block1-10', exist_ok=True)
    os.chdir('D:/Data/PSTHresults/Cruella/Block1-10')

    fig, axs = plt.subplots(8, 4, figsize=(20, 25), constrained_layout=True)

    TDTorder = [1, 3, 5, 7, 2, 4, 6, 8, 10, 12, 14, 16, 9, 11, 13, 15, 17, 19, 21, 23, 18, 20, 22, 24, 26, 28, 30, 32, 25, 27, 29, 31]
    WARPorder = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    WARPmat = np.column_stack((WARPorder, TDTorder))

    OrderofWARPeLCTRODESRCruella = np.array([[30, 31, 14, 15],
                                             [28, 29, 12, 13],
                                             [26, 27, 10, 11],
                                             [24, 25, 8, 9],
                                             [23, 22, 7, 6],
                                             [21, 20, 5, 4],
                                             [19, 18, 3, 2],
                                             [17, 16, 1, 0]])

    reordermat = []
    for itest in range(32):
        id = OrderofWARPeLCTRODESRCruella[itest, 0]
        selectrow, selectcol = np.where(WARPmat[:, 0] == id)
        reorder = TDTorder[selectrow]
        reordermat.append(reorder)
    reordermat = np.array(reordermat).reshape((8, 4)).T.flatten()
    OrderofElectrodesRWARP = np.array([17, 19, 21, 23, 24, 26, 28, 30, 16, 18, 20, 22, 25, 27, 29, 31, 1, 3, 5, 7, 8, 10, 12, 14, 0, 2, 4, 6, 9, 11, 13, 15])
    OrderofElectrodesLWARP = np.array([30, 28, 26, 24, 23, 21, 19, 17, 31, 29, 27, 25, 22, 20, 18, 16, 14, 12, 10, 8, 7, 5, 3, 1, 15, 13, 11, 9, 6, 4, 2, 0])

    for i4 in range(32):
        i3 = reordermat[i4]
        i2 = np.where(WARPmat[:, 1] == i3)[0][0]
        i22 = WARPmat[i2, 0]

        axs[i4 // 4, i4 % 4].contourf(avgspikes[:, i3].reshape((-1, 1)), freqs, lvls, data[:, i3], cmap='viridis')
        axs[i4 // 4, i4 % 4].set_title(f'TDT = {i22}, WARP = {i3}')
        axs[i4 // 4, i4 % 4].set_xlabel('Avg Spikes')
        axs[i4 // 4, i4 % 4].set_ylabel('Frequency (Hz)')

    plt.suptitle('F1815 Cruella FRA plot (March 01, 2022, Block1-10), Right Hemisphere', fontsize=16)
    plt.colorbar(axs[7, 3].images[0], ax=axs[:, -1], location='right', pad=0.01, aspect=30)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig('FRA_F1815CruellaBlock1-10.png', dpi=300)
    plt.savefig('FRA_F1815CruellaBlock1-10.svg')
    plt.savefig('FRA_F1815CruellaBlock1-10.fig')
    plt.show()