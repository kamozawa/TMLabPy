from manage import testing
from machine_learning import preprocessing
import numpy as np


class Tests(object):
    def test_zscore(self):
        a = np.random.normal(0, 1, 640)
        assert int(np.mean(preprocessing.zscore(a))) == 0

    def test_min_max(self):
        a = np.random.normal(-2, 3, 640)
        b = preprocessing.min_max(a)
        assert int(np.max(b) - np.min(b)) == 1


testing.do_test(Tests)
