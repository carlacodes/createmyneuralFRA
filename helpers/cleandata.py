import numpy as np
import matplotlib.pyplot as plt

def clean_data(action, tdata=None):
    global biglist, replacearray, orig, showData, ptspercut, useSD, xsd, highthresh, lowthresh, prepts, postpts, analogDisplayOffset, showWeights

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

    def pca(data):
        # Subtract the mean from the data
        Mn = np.mean(data, axis=0)
        data = data - Mn

        C = np.cov(data, rowvar=False)
        out = np.empty_like(data)

        for i in range(n):
            noti = np.delete(data, i, axis=1)
            Cnoti = np.delete(np.delete(C, i, axis=0), i, axis=1)

            _, v = np.linalg.eig(Cnoti)
            v = v[:, -2:]

            pc = np.dot(noti, v)
            pc = np.hstack((pc, data[:, [i]]))
            Cpc = np.cov(pc, rowvar=False)

            a1 = Cpc[0, 2] / Cpc[0, 0]
            a2 = Cpc[1, 2] / Cpc[1, 1]

            if i == 0:
                art = np.hstack((a1 * pc[:, [0]], a2 * pc[:, [1]]))
            else:
                art += np.hstack((a1 * pc[:, [0]], a2 * pc[:, [1]]))

            out[:, i] = data[:, i] - a1 * pc[:, 0] - a2 * pc[:, 1]

        art /= n
        out = np.hstack((out, art))
        return out

    def find_big_stuff(tdata):
        if useSD:
            s = np.std(tdata, axis=0)
            tdata = np.logical_or(tdata < -s * xsd, tdata > s * xsd)
        else:
            tdata = np.logical_or(tdata < lowthresh, tdata > highthresh)

        spikelist = []
        for i in range(n):
            times_on = np.where(np.diff(tdata[:, i]) == 1)[0]
            times_off = np.where(np.diff(tdata[:, i]) == -1)[0]

            if times_on.size > 0:
                times = np.column_stack((times_on, times_off))
                if times_on.size > times_off.size:
                    times[-1, 1] = tdata.shape[0] - 1
                elif times_on.size < times_off.size:
                    times = np.row_stack(([0, times_on[0]], times))

                spikelist.extend(times[:, [1, i + 1, 0]])

        return np.array(spikelist)

    def replace_big_stuff(tdata, biglist, replacearray):
        for i in range(biglist.shape[0]):
            atime, ch, btime = biglist[i]
            a = min(atime - prepts, atime - 1) if atime - prepts > 0 else 0
            b = min(btime + postpts, tdata.shape[0]) if btime + postpts < tdata.shape[0] else tdata.shape[0] - 1
            tdata[atime - a:btime + b, ch] = replacearray[atime - a:btime + b, ch]

        return tdata

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

            # if showData:
            #     ptitle = 'raw'
            #     plot_data(tdata)

            pcadata = pca(tdata)
            pca12 = pcadata[:, [-2, -1]]
            pcadata = np.delete(pcadata, [-2, -1], axis=1)
            noiseEst = tdata - pcadata
            NoSpikesData = tdata.copy()

            if showData:
                ptitle = '1st pca run'
                plot_data(pcadata)

                ptitle = 'pca1 and pca2'
                plot_data(pca12)

                ptitle = 'noise estimate'
                plot_data(noiseEst)

            biglist = find_big_stuff(pcadata)

            replacearray = np.zeros_like(tdata)
            NoSpikesData = replace_big_stuff(tdata, biglist, replacearray)
            pcadata = pca(NoSpikesData)
            pca12 = pcadata[:, [-2, -1]]
            pcadata = np.delete(pcadata, [-2, -1], axis=1)
            noiseEst = NoSpikesData - pcadata
            replacearray = noiseEst
            NoSpikesData = replace_big_stuff(tdata, biglist, replacearray)

            if showData:
                ptitle = 'spikes removed'
                plot_data(NoSpikesData)

            orig = tdata
            out = itpca(NoSpikesData)

            if showData:
                ptitle = 'final result'
                plot_data(out[:, :-2])

            return out

        elif action == 'FindBigStuff':
            return find_big_stuff(tdata)

        elif action == 'ReplaceBigStuff':
            return replace_big_stuff(tdata, biglist, replacearray)

        elif action == 'PlotData':
            plot_data(tdata)

        elif action == 'pca2':
            return pca(tdata)

    else:
        showData = tdata
        ptspercut = 24414.0625
        useSD = True
        xsd = 2.5
        highthresh = 100
        lowthresh = -100
        prepts = 10
        postpts = 10
        analogDisplayOffset = 100
        showWeights = True

        out = np.empty((0, n + 2))
        last = int(np.ceil(m / ptspercut))
        dataLength = m

        for ci in range(last):
            start = ci * ptspercut
            stop = min((ci + 1) * ptspercut, dataLength)

            cleaned_data = clean_data('GetCleanedData', data[start:stop, :])
            out = np.vstack((out, cleaned_data))

    return out, biglist


def plot_data(data):
    fig, ax = plt.subplots()
    for i in range(data.shape[1]):
        ax.plot(data[:, i] - i * analogDisplayOffset)
    plt.title('plot')
    plt.show()


# Example usage:
# cleaned_data, spikes_list = clean_data(rawdata)  # or clean_data(rawdata, 0) for no plotting, 1 for plotting
# cleaned_data contains the cleaned data, and spikes_list contains the spike information
