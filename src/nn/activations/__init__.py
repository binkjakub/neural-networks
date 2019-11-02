from src.nn.activations.hidden_activations import ReLu, Sigmoid, Tanh, Identity
from src.nn.activations.softmax import Softmax

RELU = 'relu'
SIGMOID = 'sigmoid'
TANH = 'tanh'
IDENTITY = 'identity'
SOFTMAX = 'softmax'


def get_activation(name):
    if name == RELU:
        return ReLu()
    elif name == SIGMOID:
        return Sigmoid()
    elif name == TANH:
        return Tanh()
    elif name == SOFTMAX:
        return Softmax()
    elif name == IDENTITY:
        return Identity()
    else:
        raise ValueError(f'Invalid name of activation: {name}')
