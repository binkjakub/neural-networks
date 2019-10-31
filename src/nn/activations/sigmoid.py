import numpy as np

from src.nn.activations.base import Activation


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
