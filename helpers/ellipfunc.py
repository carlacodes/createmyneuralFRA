import numpy as np
from scipy.signal import ellip, bilinear, zpk2ss, ss2zpk


def ellip_filter_design(n, Rp, Rs, Wn, btype='low', analog=False):
    if analog:
        fs = 2.0
        u = 2.0 * fs * np.tan(np.pi * Wn / fs)
    else:
        u = Wn

    if btype == 'low':
        Wn = u
    elif btype == 'band':
        Bw = u[1] - u[0]
        Wn = np.sqrt(u[0] * u[1])
    elif btype == 'high':
        Wn = u
    elif btype == 'stop':
        Bw = u[1] - u[0]
        Wn = np.sqrt(u[0] * u[1])

    z, p, k = ellip(n, Rp, Rs, Wn, analog=analog, output='zpk')

    a, b, c, d = zpk2ss(z, p, k)

    if not analog:
        a, b, c, d = bilinear(a, b, c, d, fs)

    if analog:
        return a, b, c, d
    else:
        zeros, poles, gain = ss2zpk(a, b, c, d, True)
        return b, a, zeros, poles


# Example usage:
# Design a 6th-order lowpass elliptic filter with passband edge frequency of 0.6,
# 3 dB of ripple in the passband, and 50 dB of attenuation in the stopband.
Wn = 0.6
Rp = 3.0
Rs = 50.0
n = 6

num, den, zeros, poles = ellip_filter_design(n, Rp, Rs, Wn)
print("Numerator (B):", num)
print("Denominator (A):", den)
print("Zeros:", zeros)
print("Poles:", poles)
