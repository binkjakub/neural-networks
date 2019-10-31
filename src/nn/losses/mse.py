import numpy as np

from src.nn.module import Module


class MeanSquaredError(Module):
    def __init__(self):
        super().__init__()
        self.delta = None

    def forward(self, y_true, y_pred):
        self.delta = y_true - y_pred
        return 0.5 * np.mean(np.square(self.delta))

    def backward(self, upstream_gradients):
        assert self.delta is not None
        return upstream_gradients * self.delta
