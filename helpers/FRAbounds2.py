import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

def FRAbounds(file, f32file):
    # def smoothFRA(z):
    #     s = np.array([[0.25, 0.5, 0.25],
    #                   [0.5,  1,   0.5],
    #                   [0.25, 0.5, 0.25]])
    #     s = s / np.sum(s)
    #     m, n = z.shape
    #     p = np.zeros((m + 2, n + 2))
    #     p[1:m + 1, 1:n + 1] = z
    #     p[0, :] = p[1, :]
    #     p[m + 1, :] = p[m, :]
    #     p[:, 0] = p[:, 1]
    #     p[:, n + 1] = p[:, n]
    #     z2 = np.convolve(np.convolve(p, s, mode='same'), s.T, mode='same')[1:m + 1, 1:n + 1]
    #     return z2
    def smoothFRA(z, sigma=1):
        return gaussian_filter(z, sigma=sigma)

    # def smoothFRA(z):
    #     # Define the smoothing kernel
    #     s = np.array([[0.25, 0.5, 0.25],
    #                   [0.5, 1, 0.5],
    #                   [0.25, 0.5, 0.25]])
    #
    #     # Normalize the kernel
    #     s = s / np.sum(s)
    #
    #     # Apply convolution to smooth the data
    #     smoothed_data = convolve2d(z, s, mode='same', boundary='wrap')
    #
    #     return smoothed_data
    def smooth_fra_cg(z):
        X, Y = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
        X2, Y2 = np.meshgrid(np.arange(0, z.shape[1], 0.01), np.arange(0, z.shape[0], 0.01))
        z2 = np.interp((X2.ravel(), Y2.ravel()), (X.ravel(), Y.ravel()), z.ravel()).reshape(X2.shape)
        return z2, X2, Y2


    # Load the data
    if f32file == 1:
        r = np.genfromtxt(file, delimiter=',').T
        r = r[:, :300]
        ss = np.mean(r[:, 801:1000], axis=1)
    else:
        r = file.T
        ss = file[:, 1]

    # Determine FRA parameters
    freqs = np.unique(r[1])
    levels = np.unique(r[2])
    nfreqs = len(freqs)
    nlevels = len(levels)

    # Calculate Average Spike rates
    # Initialize an empty 2D array for spike rates
    spikes = np.zeros((nlevels, nfreqs))

    for ff in range(nfreqs):
        for ll in range(nlevels):
            freq = freqs[ff]
            level = levels[ll]
            # Calculate mean spike rate for the given freq and level
            spikes[ll, ff] = np.mean(r[0][(r[1] == freq) & (r[2] == level)])

    # Smooth the FRA
    # spikes, X, X2 = smooth_fra_cg(spikes)
    spikes = smoothFRA(spikes)

    # srate = np.mean(spikes.flatten()) + (1 / 5) * np.max(spikes)
    # Calculate the threshold (srate)
    # srate = np.min(spikes)  # Use the minimum spike rate as the threshold

    mean_first_row = np.mean(spikes[0])

    # Calculate the maximum value in the entire spikes array
    max_value = np.max(spikes)

    # Calculate srate using the specified formula
    srate = mean_first_row + (1 / 5) * max_value
    # Determine boundaries of FRA
    # bounds = np.zeros(nfreqs, dtype=int)
    #
    # for ii in range(nfreqs):
    #     for jj in range(nlevels):
    #         if spikes[jj, ii] >= srate:
    #             bounds[ii] = jj
    #             break
    #
    # # Handle special cases for the highest level and no response
    # for ii in range(1, len(bounds)):
    #     if bounds[ii] == 0:
    #         bounds[ii] = nlevels
    #     if bounds[ii] == bounds[ii - 1] and bounds[ii] < nlevels:
    #         bounds[ii] = nlevels + 1

    bounds = np.zeros(nfreqs, dtype=int)

    for ii in range(nfreqs):
        response_levels = np.where(spikes[:, ii] >= srate)[0]

        if response_levels.size > 0:
            bounds[ii] = response_levels[0]
        else:
            bounds[ii] = nlevels + 1

    # Handle special cases for the highest level and no response
    for ii in range(1, len(bounds)):
        if bounds[ii] == 0:
            bounds[ii] = nlevels
        if bounds[ii] == bounds[ii - 1] and bounds[ii] < nlevels:
            bounds[ii] = nlevels + 1
    Th = (min(bounds) - 1) * 10
    a = np.where(bounds == min(bounds))[0]
    u = np.unique(freqs)
    bfs = np.log(u[a])
    # bf = np.exp(np.mean(bfs))
    bf = np.exp(np.mean(bfs)) / 1000  # Convert to kHz

    data = spikes

    # Plot the FRA
    # plt.imshow(spikes, origin='lower', aspect='auto', cmap='hot')
    # plt.xticks(np.linspace(0, spikes.shape[1] - 1, num=6), np.round(np.exp(np.linspace(np.log(min(freqs)), np.log(max(freqs)), num=6)) / 1000, 2))
    # plt.yticks(np.linspace(0, spikes.shape[0] - 1, num=6), np.round(np.linspace(min(levels), max(levels), num=6), 2))
    # plt.xlabel('Freq (kHz)')
    # plt.ylabel('Level (dB)')
    # plt.colorbar(label='Spikes per presentation')
    #
    # # Plot boundary line
    # # y_axis = np.linspace(0, spikes.shape[0] - 1, num=6)
    # plt.plot(bounds[::-1], color='white', linewidth=2)
    # plt.show()

    return bounds, bf, Th, data, spikes, levels

# Usage example:
# bounds, bf, Th, data, spikes = FRAbounds(file, f32file)
