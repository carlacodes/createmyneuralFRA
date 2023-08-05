import numpy as np
from scipy.signal import lfilter
from scipy.signal import lfilter_zi


def lpredict(x, np, npred, pos):
    if pos.lower() == 'pre':
        x = x[::-1]

    a = np.hstack([1, -np.poly(x)[:0:-1]])
    y = np.zeros(npred)
    zi = lfilter_zi(a, 1)
    for k in range(npred):
        y[k], zi = lfilter(a, 1, x[-1:-np - 1 if k >= np else -k - 1:], zi=zi)

    if pos.lower() == 'pre':
        y = y[::-1]

    return y


def filtfilthd(Hd, x, method='reflect', nfact=None):
    if not isinstance(Hd, dict):
        raise ValueError("Input not recognized. Expecting a dictionary (dfilt object).")

    if isinstance(x, np.ndarray):
        m, n = x.shape
        if m > 1 and n > 1:
            for i in range(n):
                x[:, i] = filtfilthd(Hd, x[:, i], method=method, nfact=nfact)
            return x
        else:
            x = x.flatten()
    else:
        x = np.array(x)

    len_x = len(x)

    if nfact is None:
        nfact = Hd['numtaps'] - 1

    if method.lower() not in ['reflect', 'predict', 'spline', 'pchip', 'none']:
        raise ValueError("Invalid method specified.")

    if method.lower() == 'reflect':
        nfact = min(len_x - 1, nfact)
        pre = 2 * x[0] - x[nfact::-1]
        post = 2 * x[-1] - x[-2:-nfact - 2:-1]
    elif method.lower() == 'predict':
        np = 2 * nfact
        m = np.mean(x[:np])
        pre = lpredict(x[:np] - m, nfact, nfact, 'pre') + m
        m = np.mean(x[-np:])
        post = lpredict(x[-np:] - m, nfact, nfact, 'post') + m
    elif method.lower() in ['spline', 'pchip']:
        np = 2 * nfact
        pre = np.interp(np.arange(np + 1, np + nfact + 1), np.arange(1, np + 1), x[np::-1], method=method)
        post = np.interp(np.arange(np + 1, np + nfact + 1), np.arange(1, np + 1), x[-np:], method=method)[::-1]
    else:
        pre, post = [], []

    memflag = Hd.get('persistentmemory', False)
    states = Hd.get('States', None)
    Hd['persistentmemory'] = True

    pre = lfilter(Hd['sos'], Hd['g'], pre)
    x = lfilter(Hd['sos'], Hd['g'], x)
    post = lfilter(Hd['sos'], Hd['g'], post)

    Hd['States'] = states

    post = lfilter(Hd['sos'], Hd['g'], post[::-1])[::-1]
    x = lfilter(Hd['sos'], Hd['g'], x[::-1])[::-1]

    x = x[::-1]

    Hd['States'] = states
    Hd['persistentmemory'] = memflag

    return x

# Example usage:
# x_filtered = filtfilthd(Hd, x_input)
