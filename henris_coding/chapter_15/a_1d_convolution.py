import numpy as np


def conv1d(signal, filter, p=0, s=1):
    filter_rot = np.array(filter[::-1])
    signal_padded = np.array(signal)
    if p > 0:
        zero_pad = np.zeros(shape=p)
        signal_padded = np.concatenate([zero_pad, signal_padded, zero_pad])

    res = []
    for i in range(0, int(len(signal) / s), s):
        res.append(np.sum(signal_padded[i: i + filter_rot.shape[0]] *
                          filter_rot))
    return np.array(res)


# Testing:
x = [1, 3, 2, 4, 5, 6, 1, 3]
w = [1, 0, 3, 1, 2]

print("conv1d Implementation: {}.".format(conv1d(x, w, p=2, s=1)))

print("numPy Results: {}.".format(np.convolve(x, w, mode="same")))
