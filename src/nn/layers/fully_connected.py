from src.nn.layers.linear import LinearLayer
from src.nn.module import Module


class FullyConnected(Module):
    def __init__(self, in_dim, out_dim, activation, initializer):
        super().__init__()
        self.linear = LinearLayer(in_dim, out_dim, initializer)
        self.activation = activation

    def forward(self, x):
        z = self.linear.forward(x)
        a = self.activation.forward(z)
        return a

    def backward(self, upstream_gradients):
        dZ = self.activation.backward(upstream_gradients)
        dA = self.linear.backward(dZ)
        return dA

    def parameters(self):
        return self.linear.parameters()
