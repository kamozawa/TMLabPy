import numpy as np
from numba import jit


@jit
def pse(spg: any, fmin: int = 0, fmax: int = 64):
    """Compute the spectral entropy from input spectrogram.

    Spectral entropy (SE) is a measure of the frequency distribution og the signal.
    The flatter the spectral envelope of the signal, the larger the spectral entropy.
    For example, the spectral entropy of the white noise signal is almost equals to 1.

    Parameters
    ----------
    spg: array_like
        2-D Spectrogram.

    fmin: int, optional
        The smallest frequency component of a signal. Defaults to 0.

    fmax: int, optional
        The maximum frequency component of a signal. Defaults to 64.

    Returns
    -------
    se : array_like
        Spectral entropy normalized by the maximum entropy.
    """
    se = np.zeros(spg.shape[1])
    for i in range(spg.shape[0]):
        p_k = spg[i, :] / np.sum(spg, axis=0)
        p_k = np.where((p_k == 0), 0.0001, p_k)
        se += p_k * np.log2(p_k)
    return -se / np.log2(fmax + 1 - fmin)


@jit
def pcent(spg: any, fmin: int = 0, fmax: int = 64):
    """Compute the spectral centroid of the signal from its spectrogram.

    Parameters
    ----------
    spg: array_like
        Spectrogram of the signal.

    fmin: int, optional
        The smallest frequency component of a signal. Defaults to 0.

    fmax: int, optional
        The maximum frequency component of a signal. Defaults to 64.

    Returns
    -------
    sc : array_like
        Spectral centroid of the signal.

    See Also
    --------
    pse : calculate spectral entropy from spectrogram.
    """
    res = spg.shape[-2] // fmax
    f = np.arange(spg.shape[-2], fmin+1, -res) - 1
    den = np.sum(spg[0:-1, :], axis=0)
    num = np.zeros_like(den)
    for i in range(len(num)):
        num[i] = np.sum(f*spg[0:-1, i], axis=0)
    return num/den


@jit
def pflux(spg: any):
    """Compute the spectral flux of the signal.

    Parameters
    ----------
    spg : array_like
        Spectrogram of the signal.

    Returns
    -------
    flux : array_like
        Spectral flux of the signal.
    """
    flux = np.zeros(spg.shape[-1])
    for i in range(len(flux) - 1):
        flux[i] = (np.sum(spg[:, i + 1]) - np.sum(spg[:, i])) ** 2
    return flux


@jit
def grad_change(x):
    """Compute the number of times the positive and negative of the signal's slope is inverted.

    This is different from the zero crossing rate.

    Parameters
    ----------
    x: any, array_like
        Input 1-D array.

    Returns
    -------
    count: int
        The number of times the positive and negative of the signal's slope is inverted.
    """
    count = 0
    for i in range(len(x)-1):
        if x[i+1]-x[i] < 0:
            count += 1
    return count


@jit
def zero_crossing(x: any, winsize: float = 16, overlap: float = 0.92):
    """Compute zero crossing rate.

    Parameters
    ----------
    x : array_like
        Input array.

    winsize: float, optional
        Number of samples of the analysis window size.
        Defaults to 16.

    overlap: float, optional
        Percentage of overlapping samples to window length.
        Defaults to 0.92 (92%).

    Returns
    -------
    zero_crossing_rate : array_like
        Zero crossing rate of the signal.
    """

    overlaps = int(winsize * overlap)
    zero_crossing_rate = np.zeros(int((len(x) - overlaps) / (winsize - overlaps)))
    shift = int(winsize - overlaps)

    for i in range(len(zero_crossing_rate)):

        frame = x[shift * i:shift * i + int(winsize)]
        next_frame = np.zeros_like(frame)
        next_frame[1:-1] = frame[0:-2]

        zero_crossing_rate[i] = 1/(2*len(frame)) * np.sum(np.abs(np.sign(frame)-np.sign(next_frame)))

    return zero_crossing_rate
