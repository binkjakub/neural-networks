import numpy as np

from src.nn.module import Module


class LinearLayer(Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.weights = np.random.normal(scale=0.1, size=[out_shape, in_shape])
        self.bias = np.ones(shape=[out_shape, 1])

        self.prev_activation = None
        self.z = None

        self.dW = None
        self.db = None
        self.da_prev = None

    def forward(self, x):
        self.prev_activation = x
        self.z = np.dot(self.weights, self.prev_activation) + self.bias
        return self.z

    def backward(self, upstream_gradient):
        self.dW = np.dot(upstream_gradient, self.prev_activation.T)
        self.db = np.sum(upstream_gradient, axis=1, keepdims=True)

        self.da_prev = np.dot(self.weights.T, upstream_gradient)

        # auto update
        lr = 2
        self.weights = self.weights - lr * self.dW
        self.bias = self.bias - lr * self.db
        return self.da_prev

    def _init_weights(self, method):
        pass
