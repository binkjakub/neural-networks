import numpy as np

from src.layers.module import Module


class MeanSquaredError(Module):
    def __init__(self):
        super().__init__()
        self.delta = None

    def forward(self, y_true, y_pred):
        self.delta = y_true - y_pred
        return 0.5 * np.mean(np.square(self.delta), axis=1)

    def backward(self, upstream_gradients):
        assert self.delta is not None
        return upstream_gradients * np.mean(self.delta)


class CrossEntropyLoss(Module):
    def __init__(self, network):
        super().__init__()
        self.reduction = np.mean
        self.network = network

    def forward(self, y_true, y_pred):
        return -self.reduction(y_true * np.log(y_pred + 1e-9))

    def backward(self, upstream_gradients):
        pass
