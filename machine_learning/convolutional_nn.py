from keras.models import Model, Sequential
from keras.layers import Conv1D, Conv2D, Activation, MaxPooling1D, MaxPooling2D, Flatten, Dense, Dropout


def __set_kwargs(**kwargs):
    if kwargs.get('hidden_activation') is None:
        kwargs['hidden_activation'] = 'relu'
    if kwargs.get('output_activation') is None:
        kwargs['output_activation'] = 'softmax'
    if kwargs.get('dropout') is None:
        kwargs['dropout'] = False
    if kwargs.get('droprate') is None:
        kwargs['droprate'] = .5
    return kwargs


def lenet_2d(input_shape: tuple, num_of_class: int, **kwargs):
    args = __set_kwargs(**kwargs)
    hyper_params = {'filters': [16, 32], 'kernel_size': [6, 4], 'dense': [64, 32]}

    model = Sequential()
    for n_layers in range(len(hyper_params['filters'])):
        model.add(Conv2D(filters=hyper_params['filters'][n_layers],
                         kernel_size=(hyper_params['kernel_size'][n_layers], hyper_params['kernel_size'][n_layers]),
                         strides=(1, 1), padding='same', input_shape=input_shape))
        model.add(Activation(args.get('hidden_activation')))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())

    for n_mlp in range(len(hyper_params['dense'])):
        model.add(Dense(hyper_params['dense'][n_mlp]))
        model.add(Activation(args.get('hidden_activation')))
    if args.get('dropout'):
        model.add(Dropout(args.get('droprate')))
    model.add(Dense(num_of_class))
    model.add(Activation(args.get('output_activation')))
    return model


def lenet_1d(input_shape: tuple, num_of_class: int, **kwargs):
    args = __set_kwargs(**kwargs)
    hyper_params = {'filters': [16, 32], 'kernel_size': [6, 4], 'dense': [64, 32]}

    model = Sequential()
    for n_layers in range(len(hyper_params['filters'])):
        model.add(Conv1D(filters=hyper_params['filters'][n_layers],
                         kernel_size=hyper_params['kernel_size'][n_layers],
                         strides=1, padding='same', input_shape=input_shape))
        model.add(Activation(args.get('hidden_activation')))
        model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Flatten())

    for n_mlp in range(len(hyper_params['dense'])):
        model.add(Dense(hyper_params['dense'][n_mlp]))
        model.add(Activation(args.get('hidden_activation')))
    if args.get('dropout'):
        model.add(Dropout(args.get('droprate')))
    model.add(Dense(num_of_class))
    model.add(Activation(args.get('output_activation')))
    return model
