import numpy as np
from scipy.signal import tf2ss, ss2tf

def bilinear(z, p, k, fs, fp=None, fp1=None):
    if fp is None:
        sample_freq = 2.0 * fs
    else:
        sample_freq = 2.0 * np.pi * fp / np.tan(np.pi * fp / (2.0 * fs))

    if isinstance(z, (list, tuple)):
        z = np.array(z)
    if isinstance(p, (list, tuple)):
        p = np.array(p)
    if isinstance(k, (list, tuple)):
        k = np.array(k)

    if k.size == 1 and len(z.shape) == 1 and len(p.shape) == 1:
        # Zero-pole-gain form
        zd1 = (1 + z / sample_freq) / (1 - z / sample_freq)
        zd = np.concatenate((zd1, -np.ones(p.size - zd1.size)))
        pd = (1 + p / sample_freq) / (1 - p / sample_freq)
        kd = k * np.prod(sample_freq - z) / np.prod(sample_freq - p)
        dd = None
    else:
        # Transfer function or state-space form
        if isinstance(z, (list, tuple)):
            z = np.array(z)
        if isinstance(p, (list, tuple)):
            p = np.array(p)
        if isinstance(k, (list, tuple)):
            k = np.array(k)

        if len(z.shape) == 1 and len(p.shape) == 1:
            # Transfer function form
            a, b, c, d = tf2ss(z, p)
            t = 1.0 / sample_freq
            r = np.sqrt(t)
            t1 = np.eye(a.shape[0]) + a * t / 2.0
            t2 = np.eye(a.shape[0]) - a * t / 2.0
            ad = np.linalg.solve(t2, t1)
            bd = t / r * np.linalg.solve(t2, b)
            cd = r * np.dot(c, np.linalg.solve(t2, np.eye(a.shape[0])))
            dd = np.dot(c, np.linalg.solve(t2, b)) * t / 2.0 + d
            zd, pd, kd, dd = bilinear(ad, bd, cd, dd, fs, fp1)
        else:
            # State-space form
            t = 1.0 / sample_freq
            r = np.sqrt(t)
            t1 = np.eye(z.shape[0]) + z * t / 2.0
            t2 = np.eye(z.shape[0]) - z * t / 2.0
            zd = np.linalg.solve(t2, t1)
            zd = np.dot(zd, p)  # Use the transposed form
            pd = np.dot(r, np.linalg.solve(t2, np.eye(z.shape[0])))
            kd = np.dot(k, p) / t2.diagonal().prod()
            dd = None

    return zd, pd, kd, dd

# Example usage:
# Design a 6th-order Elliptic analog low pass filter and transform
# it to a Discrete-time representation.
# Fs = 0.5  # Sampling Frequency
# z, p, k = ellipap(6, 5, 90)  # Lowpass filter prototype
# num, den = zp2tf(z, p, k)  # Convert to transfer function form
# numd, dend = bilinear(num, den, Fs)  # Analog to Digital conversion
# print("Numerator (Bd):", numd)
# print("Denominator (Ad):", dend)
