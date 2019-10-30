import numpy as np

from src.layers.module import Module


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def softmax(x):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.activation = None
        self.dZ = None

    def forward(self, x):
        self.activation = sigmoid(x)
        return self.activation

    def backward(self, upstream_grad):
        assert self.activation is not None
        self.dZ = upstream_grad * self.activation * (1 - self.activation)
        return self.dZ


class Softmax(Module):
    def __init__(self):
        super().__init__()
        self.activation = None

    def forward(self, x):
        self.activation = softmax(x)
        return self.activation

    def backward(self, upstream_grad):
        signal = self.activation
        J = - signal[..., None] * signal[:, None, :]  # off-diagonal Jacobian
        iy, ix = np.diag_indices_from(J[0])
        J[:, iy, ix] = signal * (1. - signal)  # diagonal
        grad = J.sum(axis=1)
        return upstream_grad * grad
