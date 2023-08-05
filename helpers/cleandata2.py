import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
from scipy.linalg import svd

# def pca(data):
#     Mn = np.mean(data, axis=0)
#     data = data - Mn
#
#     C = np.cov(data, rowvar=False)
#     _, v = np.linalg.eig(C)
#     v = v[:, -2:]
#
#     # Reshape data to 2D before performing matrix multiplication
#     data_reshaped = data.reshape(-1, data.shape[-1])
#     pc = np.dot(data_reshaped, v)
#     pc = pc.reshape(data.shape)  # Restore original shape
#
#     Cpc = np.cov(pc, rowvar=False)
#
#     a1 = Cpc[0, 1] / Cpc[0, 0]
#     a2 = Cpc[1, 0] / Cpc[1, 1]
#
#     art = np.sum(a1 * pc[:, :, 0], axis=1) + np.sum(a2 * pc[:, :, 1], axis=1)
#     art /= data.shape[1]
#
#     out = data - np.outer(art, v[:, 0]) - np.outer(art, v[:, 1])
#     out = np.hstack((out, np.expand_dims(art, axis=1)))
#     return out
@jit(nopython=True)
def custom_mean(data):
    m, n = data.shape
    means = np.zeros(n)
    for i in range(n):
        total = 0.0
        for j in range(m):
            total += data[j, i]
        means[i] = total / m
    return means

@jit(nopython=True)
def pca(data):
    Mn = custom_mean(data)
    data = data - Mn

    U, _, Vt = np.linalg.svd(data, full_matrices=False)
    v = Vt.T[:, -2:]

    pc = np.dot(data, v)

    Cpc = np.cov(pc, rowvar=False)

    a1 = Cpc[0, 1] / Cpc[0, 0]
    a2 = Cpc[1, 0] / Cpc[1, 1]

    art = a1 * pc[:, 0] + a2 * pc[:, 1]
    art /= data.shape[1]

    out = data - np.outer(art, v[:, 0]) - np.outer(art, v[:, 1])

    # Manual concatenation to add art as a new column to the 'out' array
    art_column = np.expand_dims(art, axis=1)
    out = np.concatenate((out, art_column), axis=1)

    return out
@jit(nopython=True)
def find_big_stuff(tdata, useSD, lowthresh, highthresh, xsd):
    m, n = tdata.shape
    spikelist = []

    for i in prange(n):
        times_on = np.where(np.diff(tdata[:, i]) == 1)[0]
        times_off = np.where(np.diff(tdata[:, i]) == -1)[0]

        if times_on.size > 0:
            times = np.empty((times_on.size, 2), dtype=np.int64)
            for j in range(times_on.size):
                start = times_on[j]
                end = times_off[j] + 1
                times[j, 0] = end
                times[j, 1] = start

            spikelist.extend(times[:, [0, i, 1]])

    return np.array(spikelist)
@jit(nopython=True)
def replace_big_stuff(tdata, biglist, replacearray, prepts, postpts):
    for i in range(biglist.shape[0]):
        atime, ch, btime = biglist[i]
        a = min(atime - prepts, atime - 1) if atime - prepts > 0 else 0
        b = min(btime + postpts, tdata.shape[0]) if btime + postpts < tdata.shape[0] else tdata.shape[0] - 1
        tdata[atime - a:btime + b, ch] = replacearray[atime - a:btime + b, ch]

    return tdata

def clean_data(action, tdata=None, ptspercut=24414.0625, useSD=True, xsd=2.5, highthresh=100, lowthresh=-100, prepts=10, postpts=10):
    global biglist, replacearray, orig, showData, analogDisplayOffset, showWeights

    if tdata is None:
        showData = 0
        data = action
        if data.shape[1] > data.shape[0]:
            data = data.T
        m, n = data.shape
        print(f"Data contains {n} channels of {m} data pts.")
    else:
        showData = tdata
        m, n = tdata.shape

    if isinstance(action, str):
        if action == 'init':
            # Edit these variables to modify the way the program cleans the data
            ptspercut = 24414.0625  # Number of data points to analyze in a 'chunk'
            useSD = True  # Use standard deviation of the analog signal to detect spikes
            xsd = 2.5  # Number of standard deviations for spike detection
            highthresh = 100  # High threshold for spike detection
            lowthresh = -100  # Low threshold for spike detection
            prepts = 10  # Start point for replacement BEFORE spike detection time
            postpts = 10  # End point for replacement AFTER RETURN UNDER (amplitude or sd) THRESHOLD
            analogDisplayOffset = 100  # Number of analog units to displace each trace when displaying
            showWeights = True  # Show weighting applied to pc vectors

        elif action == 'GetCleanedData':
            if tdata is None or tdata.size == 0:
                print('Empty rawdata. Aborting.')
                return None

            m, n = tdata.shape
            if n < 3:
                print('Rawdata must have at least 3 channels. Aborting.')
                return None
            print('calculating pca')
            pcadata = pca(tdata)
            pcadata = np.delete(pcadata, [-2, -1], axis=1)
            print('finding big stuff')
            biglist = find_big_stuff(pcadata, useSD, lowthresh, highthresh, xsd)

            replacearray = np.zeros_like(tdata)
            NoSpikesData = replace_big_stuff(tdata, biglist, replacearray, prepts, postpts)
            pcadata = pca(NoSpikesData)
            pcadata = np.delete(pcadata, [-2, -1], axis=1)
            noiseEst = NoSpikesData - pcadata
            replacearray = noiseEst
            NoSpikesData = replace_big_stuff(tdata, biglist, replacearray, prepts, postpts)

            orig = tdata
            out = pca(NoSpikesData)

            return out

        elif action == 'FindBigStuff':
            return find_big_stuff(tdata, useSD, lowthresh, highthresh, xsd)

        elif action == 'ReplaceBigStuff':
            return replace_big_stuff(tdata, biglist, replacearray, prepts, postpts)

        elif action == 'PlotData':
            plot_data(tdata)

        elif action == 'pca2':
            return pca(tdata)

    else:
        showData = tdata
        out = np.empty((0, n + 2))
        last = int(np.ceil(m / ptspercut))
        dataLength = m

        for ci in range(last):
            start = int(ci * ptspercut)
            stop = min((ci + 1) * ptspercut, dataLength)

            cleaned_data = clean_data('GetCleanedData', tdata[start:stop, :])
            out = np.vstack((out, cleaned_data))

    return out, None

def plot_data(data):
    fig, ax = plt.subplots()
    for i in range(data.shape[1]):
        ax.plot(data[:, i] - i * analogDisplayOffset)
    plt.title('plot')
    plt.show()
def plot_data(data):
    fig, ax = plt.subplots()
    for i in range(data.shape[1]):
        ax.plot(data[:, i] - i * analogDisplayOffset)
    plt.title('plot')
    plt.show()
