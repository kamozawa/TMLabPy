import numpy as np
import pywt


def multi_resolution_analysis(
        x: any,
        wname: str = 'sym4',
        level: int = 7,
        omit_level: list = None,
        mode: str = 'sym'
):
    """multi-resolution analysis from raw signal with discrete wavelet analysis.

    Parameters
    ----------
    x: any
        signal analyzed.

    wname: str, optional
        specifies the name of mother wavelet. Default to sym4.

    level: int, optional
        Specifies the decomposition level. Default to 7.

    omit_level: list, optional
        Specifies a list of decomposition levels to ignore when reconstructing the signal.

    mode: str, optional
        signal extension mode.

    Returns
    -------
    out: array_like
        Reconstructed signal.

    coefs: list
        Approximation and details coefficients.
    """
    coefs = pywt.wavedec(data=x, wavelet=wname, level=level, mode=mode)
    if omit_level is not None:
        for lv in omit_level:
            coefs[lv] = np.zeros_like(coefs[lv])
    return pywt.waverec(coeffs=coefs, wavelet=wname, mode=mode), coefs
