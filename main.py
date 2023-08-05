# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import h5py
import tdt
import pandas as pd
from helpers.cleandata2 import clean_data
import scipy.io as sio
import scipy
from helpers.getspiketimes import get_spike_times
from scipy.signal import ellip, bilinear, zpk2ss, ss2zpk
# from helpers.ellipfunc import ellip_filter_design
from helpers.filterfuncs import filtfilthd
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def highpass_filter(file_path, file_name, tank, output_folder):


    # Add paths and load the MATLAB .mat file
    # file_path = 'D:/Data/F1815_Cruella/weekfebruary282022/F1815_Cruella/'
    # file_name = 'Recording_Session_Date_01-Mar-2022_Time_15-56-19.mat'
    mat_data = scipy.io.loadmat(file_path + file_name)
    block = mat_data['recBlock']
    #
    # # Extract relevant data from the MATLAB .mat file
    # tank = 'D:/Electrophysiological Data/F1815_Cruella/'
    # block = 'Block1-10'
    #concatenate tank and block
    full_nd_path = tank + block[0]
    data = tdt.read_block(full_nd_path)
    # params = data.streams['info']
    fStim = 24414.0625*2
    fs = 24414.0625
    StartSamples = mat_data['StartSamples'].flatten()
    sTimes = StartSamples / fStim  # seconds

    # Define epoch timings and filter parameters
    sT = sTimes[:, np.newaxis] - 0.2
    sT = np.hstack((sT, sT + 1))  # epoch 400 ms before and 1.2 seconds after
    sT = (sT * fs).astype(int)  # samples
    f = np.where(sT[:, 0] > 0)[0]  # check first index is not negative
    sT = sT[f, :]
    sTseconds = sT / fs

    # Define filter
    f_order = 10
    filterWindow = [500, 3000]  # Filter cutoff frequencies
    freq_range = np.array([300, 5000]) / (fs / 2)

    # b, a = ellip(6, 0.1, 40, [300, 5000] / (fs / 2))
    Wp = np.array([300, 5000]) / (fs / 2)

    # Design the bandpass filter using ellip
    b, a = ellip(6, 0.1, 40, Wp, btype='band')

    streams = ['BB_2', 'BB_3', 'BB_4', 'BB_5']

    for i2 in range(4):
        traces = []
        for ss in range(sT.shape[0]-1):
            # Epoch and filter
            dat = data.streams[streams[i2]].data[:, sT[ss, 0]:sT[ss, 1]]
            traces_ss = [scipy.signal.filtfilt(b, a, dat[cc, :]) for cc in range(16)]
            traces.append(np.vstack(traces_ss))

        #save as matlab file as well
        sio.savemat(output_folder + f'HPf{block[0]}{i2 + 1}.mat', {'traces': traces})
        fname = f'HPf{block[0]}{i2 + 1}.h5'
        with h5py.File(output_folder + fname, 'w') as hf:
            hf.create_dataset('traces', data=traces, compression='gzip', compression_opts=9)
    return block


def clean_data_pipeline(output_folder, block, side = 'right'):
    fname = f'{output_folder}HPf{block[0]}1.h5'
    fname2 = f'{output_folder}HPf{block[0]}2.h5'

    if side == 'right':
        h = h5py.File(fname, 'r')
        hh = h5py.File(fname2, 'r')

    # Access the traces from the loaded data
    traces_h = h['traces']
    traces_hh = hh['traces']

    cleaned_data = []
    print('Cleaning data...')
    for ii in range(len(traces_hh)):
        print('at trial ' + str(ii))
        to_clean = np.vstack((traces_h[ii], traces_hh[ii]))
        cleaned_data.append(clean_data(0, to_clean))

    fs = 24414.065
    nChan = 32
    spikes = []

    for cc in range(nChan):
        spikes_in_chan = []
        for ss in range(len(traces_hh)):
            t, wv = get_spike_times(cleaned_data[ss - 1][:, cc])
            spikes_in_chan.append(t)
            spikes.append(spikes_in_chan)


    # Save the spikes data in the same format as MATLAB
    spikes_data = np.array(spikes, dtype=object)
    with h5py.File('spikes'+block[0]+side+'.h5', 'w') as f:
        dset = f.create_dataset('spikes', data=spikes, dtype=h5py.special_dtype(vlen=np.dtype('float64')))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file_path = 'D:\Data\F1815_Cruella\FRAS/'
    file_name = 'Recording_Session_Date_25-Jan-2023_Time_12-26-44.mat'
    tank = 'E:\Electrophysiological_Data\F1815_Cruella\FRAS/'
    output_folder = 'E:\Electrophysiological_Data\F1815_Cruella\FRAS\output_filtered/'

    # block = highpass_filter(file_path, file_name, tank, output_folder)
    block = ['Block1-3']
    clean_data_pipeline(output_folder, block, side = 'right')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
