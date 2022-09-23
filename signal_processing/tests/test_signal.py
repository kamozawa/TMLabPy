import numpy as np

from manage import testing
from signal_processing import signal


class Tests(object):
    def test_diff(self):
        a = [2, 3, 4]
        assert np.sum(signal.diff(a)) == 2

    def test_stft_len(self):
        a = np.random.normal(0, 1, 640)
        assert signal.stft(a).shape == (65, 313)


testing.do_test(Tests)
