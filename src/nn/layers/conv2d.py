import numpy as np
from numpy.lib.stride_tricks import as_strided

from src.nn.layers.initializers import init_parameters
from src.nn.module import Module


class Conv2d(Module):
    def __init__(self, in_dim, out_dim, initializer=None):
        super().__init__()
        self.weights, self.bias = init_parameters(in_dim, out_dim, initializer)

        self.prev_activation = None
        self.z = None

    def forward(self, x):
        """
        :param x: 4D ndarray (image) with shape: [batch_size, img_dim0, img_dim1, channels]
        :return: 4D ndarray of input convolved (SAME padding) with self.weights
        """
        padding_size = (self.weights.shape[0] - 1) // 2
        padding_axes = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
        input_ = np.pad(x, padding_axes, 'constant')
        windows = as_strided(
            input_,
            shape=(
                input_.shape[0],
                input_.shape[1] - self.weights.shape[0] + 1,
                input_.shape[2] - self.weights.shape[1] + 1,
                self.weights.shape[0],
                self.weights.shape[1],
                input_.shape[3]
            ),
            strides=(
                input_.strides[0],
                input_.strides[1],
                input_.strides[2],
                input_.strides[1],
                input_.strides[2],
                input_.strides[3]
            ),
            writeable=False
        )
        return np.einsum('xijmny, mnyk -> xijk', windows, self.weights) + self.bias

    def backward(self, upstream_gradient):
        raise NotImplementedError()

    def parameters(self):
        return [self.weights, self.bias]
