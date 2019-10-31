import numpy as np

from src.nn.layers.initializers import init_parameters
from src.nn.module import Module


class LinearLayer(Module):
    def __init__(self, in_dim, out_dim, initializer=None):
        super().__init__()
        self.weights, self.bias = init_parameters(in_dim, out_dim, initializer)

        self.prev_activation = None
        self.z = None

        self.dW = None
        self.db = None
        self.da_prev = None
        self.last_vW = np.zeros(shape=self.weights.shape)
        self.last_vb = np.zeros(shape=self.bias.shape)

    def forward(self, x):
        self.prev_activation = x
        self.z = np.dot(self.weights, self.prev_activation) + self.bias
        return self.z

    def backward(self, upstream_gradient):
        self.dW = np.dot(upstream_gradient, self.prev_activation.T)
        self.db = np.sum(upstream_gradient, axis=1, keepdims=True)

        self.da_prev = np.dot(self.weights.T, upstream_gradient)

        # auto update
        lr = 0.01
        vW = 0.9 * self.last_vW + lr * self.dW
        vb = 0.9 * self.last_vb + lr * self.db
        self.weights = self.weights - vW
        self.bias = self.bias - vb

        return self.da_prev
