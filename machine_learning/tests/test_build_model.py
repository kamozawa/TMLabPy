from manage import testing
from machine_learning import convolutional_nn


class Tests(object):
    def test_lenet2d(self):
        model = convolutional_nn.lenet_2d(input_shape=(100, 100, 1), num_of_class=2, hidden_activation='selu',
                                          output_activation='sigmoid', dropout=True, droprate=0.5)
        a = model.get_weights()[-1]
        assert a.shape[0] == 2

    def test_lenet1d(self):
        model = convolutional_nn.lenet_1d(input_shape=(100, 1), num_of_class=2, hidden_activation='selu',
                                          output_activation='sigmoid', dropout=True, droprate=0.5)
        a = model.get_weights()[-1]
        assert a.shape[0] == 2


testing.do_test(Tests)
