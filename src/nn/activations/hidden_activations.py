import numpy as np

from src.nn.activations.base import Activation


class ReLu(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.activation = np.maximum(x, 0, x)
        return self.activation

    def backward(self, upstream_grad):
        assert self.activation is not None
        self.dZ = upstream_grad * self._der_relu(self.activation)
        return self.dZ

    def _der_relu(self, x):
        return np.greater(x, 0.)


class Tanh(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.activation = np.tanh(x)
        return self.activation

    def backward(self, upstream_grad):
        assert self.activation is not None
        self.dZ = upstream_grad * (1 - self.activation ** 2)
        return self.dZ


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.activation = 1 / (1 + np.exp(-x))
        return self.activation

    def backward(self, upstream_grad):
        assert self.activation is not None
        self.dZ = upstream_grad * self.activation * (1 - self.activation)
        return self.dZ


class Identity(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.activation = x
        return self.activation

    def backward(self, upstream_grad):
        assert self.activation is not None
        self.dZ = upstream_grad
        return self.dZ
