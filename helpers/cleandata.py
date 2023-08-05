import numpy as np
from scipy.linalg import eigh

def clean_data(action, tdata=None):
    global biglist, ptitle, replacearray, orig, showData, ptspercut, useSD, xsd, highthresh, lowthresh, prepts, postpts, analogDisplayOffset, showWeights

    def pca2(tdata):
        # Subtract the mean from the tdata
        ch = tdata.shape[1]
        Mn = np.mean(tdata, axis=0)
        tdata = tdata - Mn

        C = np.cov(tdata, rowvar=False)
        out = np.zeros_like(tdata)

        for i in range(ch):
            j = np.ones(ch, dtype=bool)
            j[i] = False
            noti = tdata[:, j]  # subset of tdata not in channel i
            Cnoti = C[j][:, j]  # cov for those channels
            _, v = eigh(Cnoti)
            v = v[:, [1, 0]]  # take principle components 1 and 2
            pc = np.dot(noti, v)  # project tdata onto v
            pc = np.column_stack((pc, tdata[:, i]))  # matrix of 2 pc's and ch of interest
            Cpc = np.cov(pc, rowvar=False)  # the cov of this matrix
            a1 = Cpc[0, 2] / Cpc[0, 0]
            a2 = Cpc[1, 2] / Cpc[1, 1]
            if i == 0:
                art = np.column_stack((a1 * pc[:, 0], a2 * pc[:, 1]))
            else:
                art += np.column_stack((a1 * pc[:, 0], a2 * pc[:, 1]))

            out[:, i] = tdata[:, i] - a1 * pc[:, 0] - a2 * pc[:, 1]

        art /= ch
        out = np.column_stack((out, art))
        a1  # channel weights for first pc
        a2  # channel weights for second pc
        return out

    def itpca(no_spikes_data):
        # tdata and orig are in columns

        # Subtract the mean from the tdata
        ch = no_spikes_data.shape[1]
        M = np.mean(no_spikes_data, axis=0)
        for i in range(ch):
            no_spikes_data[:, i] -= M[i]

        C = np.cov(no_spikes_data, rowvar=False)
        out = np.zeros_like(no_spikes_data)

        for i in range(ch):
            j = np.ones(ch, dtype=bool)
            j[i] = False
            noti = no_spikes_data[:, j]  # subset of tdata not in channel i
            Cnoti = C[j][:, j]  # cov for those channels
            _, v = eigh(Cnoti)
            v = v[:, [1, 0]]  # take principal components 1 and 2
            pc = np.dot(noti, v)  # project tdata onto v
            pc = np.column_stack((pc, no_spikes_data[:, i]))  # matrix of 2 pc's and orig ch of interest
            Cpc = np.cov(pc, rowvar=False)  # the cov of this matrix
            a1 = Cpc[0, 2] / Cpc[0, 0]
            a2 = Cpc[1, 2] / Cpc[1, 1]
            if i == 0:
                art = np.column_stack((a1 * pc[:, 0], a2 * pc[:, 1]))
            else:
                art += np.column_stack((a1 * pc[:, 0], a2 * pc[:, 1]))

            out[:, i] = no_spikes_data[:, i] - a1 * pc[:, 0] - a2 * pc[:, 1]

        art /= ch
        out = np.column_stack((out, art))
        if showWeights == 'true':
            print('principle component weights for data columns 1 to', len(a1))
            print('first pc   ', ' '.join([f'{x:.3f}' for x in a1]))
            print('second pc  ', ' '.join([f'{x:.3f}' for x in a2]), '\n')
        return out

    def replace_big_stuff(tdata):
        if biglist.size == 0:
            return tdata

        replacearray = np.zeros_like(tdata)
        no_spikes_data = tdata.copy()
        for i in range(biglist.shape[0]):
            atime = biglist[i, 1]  # start time of big event
            btime = biglist[i, 2]  # stop time of big event

            a = min(atime - prepts, atime - 1)  # no out of array errors
            b = min(btime + postpts, len(tdata) - btime)  # no out of array errors

            # replace tdata points with replacearray estimate points
            no_spikes_data[atime - a:btime + b, biglist[i, 0]] = replacearray[atime - a:btime + b, biglist[i, 0]]

        return no_spikes_data

    def find_big_stuff(pcadata):
        spikelist = []
        _, n = pcadata.shape

        if useSD == 'true':
            s = np.std(pcadata, axis=0)

        for i in range(n):
            times = []
            if useSD == 'true':
                pcadata[:, i] = (pcadata[:, i] < -s[i] * xsd) | (pcadata[:, i] > s[i] * xsd)
            else:
                pcadata[:, i] = (pcadata[:, i] < lowthresh) | (pcadata[:, i] > highthresh)

            times = np.where(np.diff(pcadata[:, i]) == 1)[0]  # find the times of the first ones
            times2 = np.where(np.diff(pcadata[:, i]) == -1)[0]  # find the times of the last ones

            if times.size > 0:
                times = times[:, None]
                times = np.hstack((times, i * np.ones((times.shape[0], 1), dtype=int)))  # (start time, channel)
                if len(times2) == len(times):
                    times = np.hstack((times, times2[:, None]))  # (start time, channel, stop time)
                elif len(times2) == len(times) - 1:
                    # End of data in the middle of a big spike
                    times[:-1, 2] = times2
                    times[-1, 2] = tdata.shape[0] - 1  # Last point is the end of spike
                else:
                    # Beginning of data in a big spike
                    times = np.vstack(([0, times2[0], 0], times))  # (start time, channel, stop time)

                spikelist.extend(times.tolist())  # put stuff in the right column and add to running total

        spikelist = np.array(spikelist, dtype=int)
        return spikelist

    def get_cleaned_data(data_chunk):
        m, n = data_chunk.shape

        if showData:
            ptitle = 'raw'
            plot_data(data_chunk)

        # Step 2: PCA the data_chunk
        pcadata = pca2(data_chunk)  # get 1st cleaned estimate
        pca12 = pcadata[:, [pcadata.shape[1] - 2, pcadata.shape[1] - 1]]  # last two columns are pca 1 and pca2
        pcadata = pcadata[:, :-2]  # get rid of them for now
        noiseEst = data_chunk - pcadata  # get noise estimate
        no_spikes_data = data_chunk.copy()  # copy of tdata for noise replacement

        if showData:
            ptitle = '1st pca run'
            plot_data(pcadata)

            ptitle = 'pca1 and pca2'
            plot_data(pca12)

            ptitle = 'noise estimate'
            plot_data(noiseEst)

        # Step 3: Get a list of putative spikes in tdata
        biglist = find_big_stuff(pcadata)

        # Step 4: Replace the spikes with the noise estimate
        # First use zeros to replace spikes to avoid cross contamination of noise estimate,
        # then replace spikes with noise estimate
        replacearray = np.zeros_like(data_chunk)
        no_spikes_data = replace_big_stuff(data_chunk)
        pcadata = pca2(no_spikes_data)  # get 1st cleaned estimate
        pca12 = pcadata[:, [pcadata.shape[1] - 2, pcadata.shape[1] - 1]]  # last two columns are pca 1 and pca2
        pcadata = pcadata[:, :-2]  # get rid of them for now
        noiseEst = no_spikes_data - pcadata  # get noise estimate
        replacearray = noiseEst
        no_spikes_data = replace_big_stuff(data_chunk)  # now replace spikes with noise estimate

        if showData:
            ptitle = 'spikes removed'
            plot_data(no_spikes_data)

        # Step 5: Do second order cleaning
        orig = data_chunk
        out = itpca(no_spikes_data)

        if showData:
            ptitle = 'final result'
            plot_data(out[:, :-2])

        return out

    def plot_data(data_chunk):
        import matplotlib.pyplot as plt
        m, n = data_chunk.shape

        for i in range(n):
            data_chunk[:, i] = data_chunk[:, i] - i * analogDisplayOffset

        plt.plot(data_chunk)
        plt.title(ptitle)
        plt.show()

    if action == 'init':
        # Edit these variables to modify the way the program cleans the data
        # Unless your conditions differ considerably, we recommend starting with the values given here.
        ptspercut = 24414.0625  # Number of data points to analyze in a 'chunk'
        useSD = 'true'  # Use standard deviation of the analog signal to detect spikes
        xsd = 2.5  # Number of standard deviations that make something a putative spike
        highthresh = 100  # Used only if useSD is not 'true'; high and low are 'or'ed
        lowthresh = -100  # Used only if useSD is not 'true'; high and low are 'or'ed
        prepts = 10  # Start point for replacement BEFORE spike detection time
        postpts = 10  # End point for replacement AFTER RETURN UNDER (amplitude or sd) THRESHOLD
        analogDisplayOffset = 100  # Number of analog units by which to displace each trace when displaying
        showWeights = 'true'  # Show weighting applied to pc vectors
        return

    if isinstance(action, str):
        if action == 'GetCleanedData':
            if tdata is None or tdata.size == 0:
                return np.array([])
            m, n = tdata.shape

            if n < 3:
                return np.array([])

            global ptitle
            showData = tdata
            ptitle = 'raw'
            return get_cleaned_data(tdata)
        elif action == 'FindBigStuff':
            # Get cleaned data using PCA
            pcadata = pca2(tdata)
            return find_big_stuff(pcadata)
        elif action == 'ReplaceBigStuff':
            return replace_big_stuff(tdata)
        elif action == 'PlotData':
            plot_data(tdata)
            return
        elif action == 'pca2':
            return pca2(tdata)
        elif action == 'itpca':
            return itpca(tdata)
    else:
        showData = tdata if tdata is not None else 0
        clean_data('init')
        out = np.array([])

        if isinstance(action, np.ndarray):
            data = action
            m, n = data.shape

            if n > m:
                data = data.T
                m, n = data.shape

            print(f'Data contains {n} channels of {m} data points.')

            last = int(np.ceil(m / ptspercut))  # analyze data in ptspercut pieces
            dataLength = m

            for ci in range(last):
                if (ci + 1) * ptspercut > dataLength:
                    stop = dataLength  # last piece is whatever is left
                else:
                    stop = (ci + 1) * ptspercut

                cleaned_data = get_cleaned_data(data[ci * ptspercut:stop, :])
                out = np.vstack((out, cleaned_data)) if out.size != 0 else cleaned_data

        return out


# Example usage:
# cleaned_data = clean_data(raw_data)  # Clean the data without plotting
# cleaned_data = clean_data(raw_data, 0)  # Clean the data without plotting
# cleaned_data = clean_data(raw_data, 1)  # Clean the data with plotting
