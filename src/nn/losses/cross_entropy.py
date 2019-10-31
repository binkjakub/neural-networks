import numpy as np

from src.nn.module import Module


class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        self.delta = y_true - y_pred
        return -np.mean(y_true * np.log(y_pred + 1e-9))

    def backward(self, upstream_gradient=1):
        return upstream_gradient * self.delta


class CrossEntropyWithLogitLoss(Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, y_true, y_pred):
        return

    def backward(self, upstream_gradients):
        return
