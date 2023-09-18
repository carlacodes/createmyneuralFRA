# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import h5py
import tdt
import pickle
import pandas as pd
from helpers.cleandata2 import clean_data
import scipy.io as sio
import scipy
from helpers.plotfrafromspikes import run_fra
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
    mat_data = pd.read_csv(file_path + file, delimiter='\t')
    # recblock is in the name of the file
    block = file.split()[3]
    # remove the .txt
    block = block[:-4]

    #
    # # Extract relevant data from the MATLAB .mat file
    # tank = 'D:/Electrophysiological Data/F1815_Cruella/'
    # block = 'Block1-10'
    #concatenate tank and block
    full_nd_path = tank + block
    try:
        data = tdt.read_block(full_nd_path)
    except:
        print('error reading block')
        print(full_nd_path)
        return block
        # params = data.streams['info']
    fStim = 24414.0625*2
    fs = 24414.0625
    StartSamples = mat_data['StartTime']
    sTimes = StartSamples  # seconds
    #convert sTimes to numpy
    sTimes = np.array(sTimes)
    # Define epoch timings and filter parameters
    sT = sTimes[:, np.newaxis] - 0.2
    sT = np.hstack((sT, sT + 1))  # epoch so total duration is 0.8s
    sT = (sT * fs).astype(int)  # samples
    f = np.where(sT[:, 0] > 0)[0]  # check first index is not negative
    sT = sT[f, :]
    f = np.where(sT[:, 1] < data.streams['BB_2'].data.shape[1])[0]  # check last index is not larger than data
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

    streams = ['BB_2', 'BB_3']


    for i2 in range(len(streams)):
        traces = []
        for ss in range(sT.shape[0] - 1):
            # Epoch and filter
            try:
                dat = data.streams[streams[i2]].data[:, sT[ss, 0]:sT[ss, 1]]
            except:
                print('error reading stream')
                print(streams[i2])
                return block
            traces_ss = [scipy.signal.filtfilt(b, a, dat[cc, :]) for cc in range(16)]
            traces.append(np.vstack(traces_ss))

        # Remove elements with inhomogeneous shapes
        reference_shape = traces[0].shape[1]
        drop_list = [ii for ii, trace in enumerate(traces) if trace.shape[1] != reference_shape]
        traces = [trace for ii, trace in enumerate(traces) if ii not in drop_list]

        # Save traces to a .h5 file
        fname = f'HPf{block[0]}{i2 + 1}.h5'
        with h5py.File(output_folder + fname, 'w') as hf:
            hf.create_dataset('traces', data=np.array(traces), compression='gzip', compression_opts=9)

    return block


def clean_data_pipeline(output_folder, block, side = 'right'):


    if side == 'right':
        fname = f'{output_folder}HPf{block[0]}1.h5'
        fname2 = f'{output_folder}HPf{block[0]}2.h5'

    elif side == 'left':
        fname = f'{output_folder}HPf{block[0]}3.h5'
        fname2 = f'{output_folder}HPf{block[0]}4.h5'

    # Access the traces from the loaded data
    try:
        h = h5py.File(fname, 'r')
        hh = h5py.File(fname2, 'r')
        traces_h = h['traces']
        traces_hh = hh['traces']

    except:
        print('error reading file')
        print(fname)
        return block

    cleaned_data = []
    print('Cleaning data...')
    try:
        for ii in range(len(traces_hh)):
            print('at trial ' + str(ii))
            to_clean = np.vstack((traces_h[ii], traces_hh[ii]))

            cleaned_data.append(clean_data(0, to_clean))
    except:
        print('error cleaning data, probably shitty data')
        print(fname)
        return block

    fs = 24414.065
    nChan = 32
    spikes = []
    # spikes = np.empty((nChan, len(traces_hh)))
    #spikes needs to be a nChan by nTrials cell array
    # spikes= np.empty((nChan, len(traces_hh)))
    for cc in range(nChan):
        spikes_in_chan = []
        for ss in range(len(traces_hh)):
            # test = cleaned_data[ss]
            # test2=test[0][:,cc]
            # print('getting spike times for trial ' + str(ss) + ' channel ' + str(cc) )
            #get the spike times -0.1s to 0.1s around the stim
            #assuming each trial is 1s long, so 0.2s around the stim
            start = int((0.1)*fs)  # 0.1s before stim
            end = int((0.3*fs)) # 0.1s after stim
            t, wv = get_spike_times(cleaned_data[ss][0][cc,:])
            # t, wv = get_spike_times(cleaned_data[ss][0][cc,:])
            spikes_in_chan.append(t)


        # spikes[cc,:] = spikes_in_chan
        spikes.append(spikes_in_chan)

    # Save the spikes data in the same format as MATLAB
    spikes_data = np.array(spikes, dtype=object)
    #check spikes is not empty
    if spikes_data.size == 0:
        print('No spikes found!')
    print('saving spikes' + block[0] + side + '.h5')
    #save as pickle file
    with open(output_folder + 'spikes' + block[0] + side + '.pkl', 'wb') as f:
        pickle.dump(spikes_data, f)
    # #save to output folder
    # with h5py.File(output_folder + 'spikes' + block[0] + side + '.h5', 'w') as f:
    #     dset = f.create_dataset('spikes', data=spikes_data, dtype=h5py.special_dtype(vlen=np.dtype('float64')))
    # with h5py.File('spikes'+block[0]+side+'.h5', 'w') as f:
    #     dset = f.create_dataset('spikes', data=spikes, dtype=h5py.special_dtype(vlen=np.dtype('float64')))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # file_name = 'Recording_Session_Date_25-Jan-2023_Time_12-26-44.mat'
    tank = 'E:\Electrophysiological_Data\F1306_Firefly/'
    output_folder = 'E:\Electrophysiological_Data\F1306_Firefly/FRAS/'

    file_path = 'D:\Data\F1306_Firefly\F1306_Firefly\FRAS/'
    #get a lsit of all the files in the directory
    import os
    files = os.listdir(file_path)

    #exclude all files that don't end with .mat
    files = [file for file in files if file.endswith('.txt')]
    #only the right side good for zola
    # files = ['Recording_Session_Date_09-Mar-2020_Time_14-17-40.mat']
    for file in files:
        print(file)
        #load a txt file
        #read a .txt. file instead of a mat data file
        mat_data = pd.read_csv(file_path + file, delimiter='\t')
        # recblock is in the name of the file
        block = file.split()[3]
        # remove the .txt
        block = block[:-4]



        #
        # block = mat_data['recBlock']
        # #
        # block = highpass_filter(file_path, file, tank, output_folder)
        #
        # # block = highpass_filter(file_path, file, tank, output_folder)
        # #
        # # # print(block)
        # clean_data_pipeline(output_folder, block, side = 'right')

        run_fra('right', file_path, file, output_folder, animal = 'F1306')
        # run_fra('left', file_path, file, output_folder)



    # clean_data_pipeline(output_folder, block, side = 'left')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
