# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import h5py
import tdt
import scipy
from helpers.ellipfunc import ellip_filter_design
from helpers.filterfuncs import filtfilt
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def highpass_filter():


    # Add paths and load the MATLAB .mat file
    file_path = 'D:/Data/F1815_Cruella/weekfebruary282022/F1815_Cruella/'
    file_name = 'Recording_Session_Date_01-Mar-2022_Time_15-56-19.mat'
    mat_data = scipy.io.loadmat(file_path + file_name)

    # Extract relevant data from the MATLAB .mat file
    tank = 'D:/Electrophysiological Data/F1815_Cruella/'
    block = 'Block1-10'

    data = tdt.read_block(tank, block)
    params = data.streams['info']
    fStim = params['fs']
    fs = 24414.0625
    StartSamples = data.streams['StartSamples'].data.flatten()
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

    b, a = ellip_filter_design(6, 0.1, 40, [300, 5000] / (fs / 2))

    streams = ['BB_2', 'BB_3', 'BB_4', 'BB_5']
    output_folder = 'D:/Electrophysiological Data/F1815_Cruella/HP_Block1-10/'

    for i2 in range(4):
        traces = []
        for ss in range(1096 - 1):
            # Epoch and filter
            dat = data.streams[streams[i2]].data[:, sT[ss, 0]:sT[ss, 1]]
            traces_ss = [filtfilt(b, a, dat[cc, :]) for cc in range(16)]
            traces.append(np.vstack(traces_ss))

        fname = f'HPfBlock1-10{i2 + 1}.h5'
        with h5py.File(output_folder + fname, 'w') as hf:
            hf.create_dataset('traces', data=traces, compression='gzip', compression_opts=9)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
