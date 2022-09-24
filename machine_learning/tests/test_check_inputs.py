from manage import testing
from machine_learning import preprocessing
import numpy as np


class Tests(object):
    def test_check_array(self):
        a = np.vstack([np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5])])
        preprocessing.check_array(a)


testing.do_test(Tests)
