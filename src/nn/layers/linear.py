import numpy as np

from src.nn.layers.initializers import init_parameters
from src.nn.module import Module


class LinearLayer(Module):
    def __init__(self, in_dim, out_dim, initializer=None):
        super().__init__()
        self.weights, self.bias = init_parameters(in_dim, out_dim, initializer)

        self.prev_activation = None
        self.z = None

    def forward(self, x):
        self.prev_activation = x
        self.z = np.dot(self.weights.val, self.prev_activation) + self.bias.val
        return self.z

    def backward(self, upstream_gradient):
        self.weights.grad = np.dot(upstream_gradient, self.prev_activation.T)
        self.bias.grad = np.sum(upstream_gradient, axis=1, keepdims=True)

        grad_act_prev = np.dot(self.weights.val.T, upstream_gradient)
        return grad_act_prev

    def parameters(self):
        return [self.weights, self.bias]
