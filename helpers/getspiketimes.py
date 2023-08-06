import numpy as np
from scipy import interpolate

def get_spike_times(x):
    # Parameters
    # wInt = 1
    # interpFactor = 4
    #
    # interpInt = wInt / interpFactor
    # window = np.arange(-15, 18, wInt)
    # interpWind = np.arange(-15, 18, interpInt)
    wInt = 1.0  # Use floating-point number for wInt
    interpFactor = 4

    interpInt = wInt / interpFactor
    window = np.linspace(-15, 17, int((17 - (-15)) / wInt) + 1)  # Use np.linspace for strictly increasing array
    interpWind = np.linspace(-15, 17, int((17 - (-15)) / interpInt) + 1)  # Use np.linspace for interpWind

    nW = len(window) + 1
    alignmentZero = np.where(window == 0)[0][0]

    # Preassign
    t, wv = [], []

    # Format
    x = np.array(x)
    if x.ndim == 2:
        x = np.squeeze(x)

    # Get upper (ub) and lower (lb) bounds
    lb = -3.5 * np.nanstd(x)
    ub = -8 * np.nanstd(x)

    # Identify threshold crossings
    lcIdx = np.where(x < lb)[0]
    ucIdx = np.where(x < ub)[0]

    # Remove events exceeding the upper threshold
    lcIdx = np.setdiff1d(lcIdx, ucIdx)

    # Move to next trial if no events were found
    if len(lcIdx) == 0:
        return [], []

    # Identify crossing points in samples
    crossThreshold = lcIdx[np.where(np.diff(lcIdx) != 1)[0]]

    # Remove events where window cannot fit
    crossThreshold = crossThreshold[(crossThreshold >= nW) & (crossThreshold <= len(x) - nW)]

    # Get interim waveforms
    wvIdx = crossThreshold[:, np.newaxis] + window[np.newaxis, :]
    #convert wvIdx to int
    wvIdx = wvIdx.astype(int)
    wv = x[wvIdx]

    # Move to next trial if no waveforms are valid
    if len(wv) == 0:
        return [], []

    # Remove any waveforms that contain nans
    rm_idx = np.sum(np.isnan(wv), axis=1) > 0
    crossThreshold = crossThreshold[~rm_idx]
    wv = wv[~rm_idx]

    # Interpolate waveforms
    interp = interpolate.interp1d(window, wv, kind='cubic', axis=1)
    wv = interp(interpWind)

    # Align events
    peakIdx = np.argmin(wv, axis=1)
    peakIdx = np.round(peakIdx / interpFactor).astype(int)
    alignmentShift = peakIdx - alignmentZero
    alignedCrossings = crossThreshold + alignmentShift

    # Reset events where window cannot fit (i.e. don't throw away, just include without alignment)
    alignedCrossings = np.where(alignedCrossings < nW, crossThreshold, alignedCrossings)
    alignedCrossings = np.where(alignedCrossings > (len(x) - nW), crossThreshold, alignedCrossings)

    return alignedCrossings, crossThreshold
