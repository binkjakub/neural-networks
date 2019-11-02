import numpy as np

from src.nn.activations.softmax import softmax
from src.nn.module import Module


class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()
        self.delta = None

    def forward(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred + 1e-9))

    def backward(self, upstream_gradient=1):
        return upstream_gradient

    def parameters(self):
        return None


class CrossEntropyWithLogitLoss(Module):
    def __init__(self):
        super().__init__()
        self.class_proba = None
        self.y_true = None

    def forward(self, y_true, y_pred):
        self.class_proba = softmax(y_pred)
        self.y_true = y_true.astype(bool)

        correct_log_proba = -np.log(self.class_proba[self.y_true])
        return np.sum(correct_log_proba) / y_true.shape[1]

    def backward(self, upstream_gradients=1):
        dZ = np.where(self.y_true, self.class_proba - 1, self.class_proba)
        dZ = dZ / self.class_proba.shape[1]
        return upstream_gradients * dZ

    def parameters(self):
        return None
