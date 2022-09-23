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
