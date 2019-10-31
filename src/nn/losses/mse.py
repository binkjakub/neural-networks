import numpy as np

from src.nn.module import Module


class MeanSquaredError(Module):
    def __init__(self):
        super().__init__()
        self.delta = None

    def forward(self, y_true, y_pred):
        self.delta = y_true - y_pred
        return 0.5 * np.mean(np.square(self.delta))

    def backward(self, upstream_gradients=1):
        assert self.delta is not None
        return upstream_gradients * (-1 / self.delta.shape[1]) * self.delta

    def parameters(self):
        return None
