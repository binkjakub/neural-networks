import numpy as np

from src.nn.activations.base import Activation


class Softmax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        shift_x = x - np.max(x)
        exp_x = np.exp(shift_x)
        self.activation = exp_x / np.sum(exp_x)
        return self.activation

    def backward(self, upstream_grad):
        assert self.activation is not None
        signal = self.activation
        J = - signal[..., None] * signal[:, None, :]  # off-diagonal Jacobian
        iy, ix = np.diag_indices_from(J[0])
        J[:, iy, ix] = signal * (1. - signal)  # diagonal
        self.dZ = J.sum(axis=1)
        return upstream_grad * self.dZ
