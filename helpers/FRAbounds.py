import numpy as np
import matplotlib.pyplot as plt
#
# def smooth_fra(z):
#     # Do some smoothing
#     s = np.array([[0.25, 0.5, 0.25],
#                   [0.5, 1, 0.5],
#                   [0.25, 0.5, 0.25]])
#     s = s / np.sum(s)  # normalize the window
#     m, n = z.shape
#     p = np.zeros((m + 2, n + 2))
#     p[1:m + 1, 1:n + 1] = z
#     p[0, :] = p[1, :]
#     p[m + 1, :] = p[m, :]
#     p[:, 0] = p[:, 1]
#     p[:, n + 1] = p[:, n]
#     z2 = np.convolve(p.flatten(), s.flatten(), 'valid').reshape((m, n))
#     return z2
def smooth_fra(z):
    # Create a sliding window (kernel)
    s = np.array([[0.25, 0.5, 0.25],
                  [0.5, 1, 0.5],
                  [0.25, 0.5, 0.25]])

    s = s / s.sum()  # Normalize the kernel

    m, n = z.shape
    p = np.zeros((m + 2, n + 2))
    p[1:m + 1, 1:n + 1] = z  # Zero-padding to maintain the original size

    # Perform convolution and reshape the result
    z2 = np.convolve(p.flatten(), s.flatten(), 'valid').reshape((m, n))
    return z2

def genFRAbounds(file, f32file):
    fsz = 10
    if f32file == 1:
        r = spikerate(file, 1, 200, 1, 2)
        r = r.T
        ss = spikerate(file, 801, 1000, 1, 2)
        ss = ss.T
    else:
        r = file

    freqs = np.unique(r[:, 1])
    levels = np.unique(r[:, 2])
    nfreqs = len(freqs)
    nlevels = len(levels)

    spikes = np.zeros((nlevels, nfreqs))

    for ff in range(nfreqs):
        for ll in range(nlevels):
            spikes[ll, ff] = np.mean(r[(r[:, 1] == freqs[ff]) & (r[:, 2] == levels[ll]), 0])

    # Apply smoothing to the FRA
    # spikes, X, X2 = smooth_fra(spikes)

    # Calculate the mean spontaneous rate
    srate = np.mean(spikes[0, :]) + (1 / 5) * np.max(spikes)

    # Determine boundaries of FRA
    bounds = np.zeros(nfreqs)
    for ii in range(nfreqs):
        for jj in range(nlevels):
            if spikes[jj, ii] < srate and jj < nlevels:
                # no response, move on to the next level
                pass
            elif spikes[jj, ii] >= srate and jj == 0:
                # response at the lowest level, move onto the next to see if real
                pass
            elif spikes[jj, ii] >= srate and 1 <= jj < nlevels - 2 and \
                    spikes[jj - 1, ii] >= srate and spikes[jj + 1, ii] >= srate and spikes[jj + 2, ii] > srate:
                bounds[ii] = jj - 1
                break
            elif spikes[jj, ii] >= srate and jj == nlevels - 2 and spikes[jj - 1, ii] >= srate and spikes[jj + 1, ii] >= srate:
                bounds[ii] = jj - 1
                break
            elif spikes[jj, ii] >= srate and jj == nlevels - 1 and spikes[jj - 1, ii] >= srate:
                bounds[ii] = jj - 1
                break
            elif spikes[jj, ii] >= srate and jj == nlevels - 1:
                bounds[ii] = nlevels
            elif jj == nlevels - 1 and spikes[jj, ii] < srate:
                bounds[ii] = nlevels + 1

    for ii in range(1, len(bounds) - 1):
        if bounds[ii - 1] == nlevels + 1 and bounds[ii + 1] == nlevels + 1 and bounds[ii] < 4:
            bounds[ii] = nlevels + 1

    bf_idx = np.argwhere(bounds == np.min(bounds) - 1)
    u = np.unique(freqs)
    bfs = np.log(u[bf_idx])
    bf = np.exp(np.mean(bfs))
    data = spikes
    return bounds, bf, srate, data, spikes

# Example usage:
# bounds, bf, Q10, Q30, Th, spikes = FRAbounds(file, f32file)
# Here, file should be a 3 column array with mean spike rate, frequencies, and levels.
# f32file should be set to 1 if file is in f32 format, or 0 if file is in Src format.
# The function returns the boundaries, best frequency (bf), Q10 and Q30, threshold (Th), and smoothed spike data.

# You may need to implement the spikerate function separately, as it's not provided in the code snippet you shared.
