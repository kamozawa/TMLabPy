import numpy as np


def zscore(x: any, axis=None):
    """Compute the z score.

    Compute the z score of each value in the sample,
    relative to the sample mean and standard deviation.

    Parameters
    ----------
    x: any, array_like
        An array like object containing the sample data.

    axis: int or None, optional
        Axis along which to operate. Default is 0. If None, compute over the whole array a.

    Returns
    -------
    zs : array_like
        The z-scores, standardized by mean and standard deviation of input array x.

    See Also
    --------
    min_max: normalized array.

    Notes
    -----
    This function preserves ndarray subclasses, and works also with matrices and masked arrays
    (it uses asanyarray instead of asarray for parameters).
    """
    zs = (x-x.mean(axis=axis, keepdims=True))/np.std(x, axis=axis, keepdims=True)
    return zs


def min_max(x: any, axis=None):
    """Transform features by scaling each feature to the range of 0 to 1.

    Parameters
    ----------
    x: any, array_like
        The data.

    axis: int or None, optional
        Axis used to scale along. If 0, independently scale each feature, otherwise (if 1) scale each sample.

    Returns
    -------
    result : array_like
        The array, normalized by max and min value of input array x.

    See Also
    --------
    zscore: Compute the z score.
    """
    result = (x-x.min(axis=axis, keepdims=True))/(x.max(axis=axis, keepdims=True)-x.min(axis=axis, keepdims=True))
    return result


def seqseg(
        x: any,
        length: int,
        fs: int = 128
) -> np.array:
    """Returns a two-dimensional array of time-series signals divided into any lengths.

    Parameters
    ----------
    x: any, array_like
        Input array sequence.

    length: int or float
        Segment length to divide. Must be shorter than input.
        Unit is second.

    fs: int, optional
        Sampling frequency of the signal. Defaults to 128.

    Returns
    -------
    out : array_like
        Array segmented into the shape of 2D.

    Examples
    --------
    >>> a = np.zeros(100)
    ... a.shape
    (100,)
    >>> b = seqseg(x=a, length=10, fs=1)
    ... b.shape
    (10, 10)
    """
    num = len(x) // (length*fs)
    itr = map(lambda n: x[int(length*fs)*n:int(length*fs)*(n+1)],
              range(num))
    return np.array([i for i in itr])
