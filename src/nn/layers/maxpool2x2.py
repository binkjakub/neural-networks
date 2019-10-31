import numpy as np
from numpy.lib.stride_tricks import as_strided

from src.nn.layers.initializers import init_parameters
from src.nn.module import Module


class MaxPool2x2(Module):
    def __init__(self, in_dim, out_dim, initializer=None):
        super().__init__()
        self.weights, self.bias = init_parameters(in_dim, out_dim, initializer)

        self.prev_activation = None
        self.z = None

    def forward(self, x):
        """
        :param x: 4D tensor with shape: [batch_size, features_dim0, features_dim1, channels]
        :return: 4D tensor max-pooled with constraints: window=2x2, strides=(2,2), padding SAME
        """
        windows = as_strided(
            x,
            shape=(
                x.shape[0],
                x.shape[1] // 2,
                x.shape[2] // 2,
                2,
                2,
                x.shape[3]
            ),
            strides=(
                x.strides[0],
                2 * x.strides[1],
                2 * x.strides[2],
                x.strides[1],
                x.strides[2],
                x.strides[3]
            ),
            writeable=False
        )
        return np.amax(windows, axis=(3, 4))

    def backward(self, upstream_gradient):
        raise NotImplementedError()

    def parameters(self):
        return [self.weights, self.bias]
