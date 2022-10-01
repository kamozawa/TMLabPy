import numpy as np


def diff(x: any):
    """Differentiate an array by numeric difference.

    Parameters
    ----------
    x: array_like
        Input array.

    Returns
    -------
    dif : array_like
        Differentiated array.
    """
    dif = [x[i+1]-x[i] for i in range(len(x)-1)]
    return np.array(dif)


def stft(
        x: any,
        winsize: float = 16,
        overlap: float = 0.92,
        nfft: int = 128
):
    """Calculate a spectrogram with short-time Fourier transform (STFT).

    Spectrograms can be used as a way of visualizing
    the change of a non-stationary signal’s, such as ECG, frequency content over time.

    Parameters
    ----------
    x: array_like
        Input 1-D array.

    winsize: float, optional
        Number of samples of the analysis window size.
        Defaults to 16.

    overlap: float, optional
        Percentage of overlapping samples to window length.
        Defaults to 0.92 (92%).

    nfft: int, optional
        Length of the transformed axis of the output.
        If n is smaller than the length of the input, the input is cropped.
        Defaults to 128.

    Returns
    -------
    spg : array_like
        Calculated spectrogram.

    See Also
    --------
    numpy.fft.fft : Compute the one-dimensional discrete Fourier Transform.
    scipy.signal.spectrogram : Compute a spectrogram with consecutive Fourier transforms.
    """
    assert winsize < len(x)

    overlaps = int(winsize*overlap)
    spg = np.zeros((int(nfft/2+1),
                    int((len(x)-overlaps)/(winsize-overlaps))))
    shift = int(winsize-overlaps)

    for j in range(len(spg[0, :])):
        frame = x[shift*j:shift*j+int(winsize)]
        frame = np.hamming(int(winsize)) * frame

        amp = np.fft.fft(frame, nfft)
        amp = amp[0:int(len(amp)/2+1)]
        spg[:, j] = np.abs(amp)

    return spg


def ssa(x, window_size: int = 50, do_normalize: bool = True):
    """calculates abnormality by singular spectral analysis (SSA).

    Parameters
    ----------
    x:
        time series data.

    window_size: int, optional
        window (test) size of test matrix. Default to 50.

    do_normalize: bool, optional
        If True, calculated abnormality is normalized by its maximum value.

    Returns
    -------
    score: ndarray
        Abnormality calculated by SSA.
    """

    def __embed(lst, dim):
        emb = np.empty((0, dim), float)
        for i in range(lst.size - dim + 1):
            tmp = np.array(lst[i:i + dim]).reshape((1, -1))
            emb = np.append(emb, tmp, axis=0)
        return emb

    k = window_size // 2
    lag = k // 2  # lag, corresponds shift width

    score = np.zeros_like(x)
    for t in range(window_size + k, len(x) - lag + 1 + 1):
        t_start = t - window_size - k + 1
        t_end = t - 1
        x1 = __embed(x[t_start:t_end], window_size).T[::-1, :]
        x2 = __embed(x[(t_start + lag):(t_end + lag)], window_size).T[::-1, :]  # Test mtx.

        u1, s1, v1 = np.linalg.svd(x1, full_matrices=True)
        u1 = u1[:, 0:2]
        u2, s2, v2 = np.linalg.svd(x2, full_matrices=True)
        u2 = u2[:, 0:2]
        u, s, v = np.linalg.svd(u1.T.dot(u2), full_matrices=True)

        score[t] = 1 - np.square(s[0])

    if do_normalize:
        score /= np.max(score)
    return score
