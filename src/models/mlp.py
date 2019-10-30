from src.layers.activations import Softmax, Sigmoid
from src.layers.layers import (FullyConnected)
from src.layers.module import Module


class MultiLayerPerceptron(Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.hidden = FullyConnected(in_shape=input_shape,
                                     out_shape=256,
                                     activation=Sigmoid())

        self.output = FullyConnected(in_shape=256,
                                     out_shape=output_shape,
                                     activation=Softmax())

    def forward(self, x):
        result = self.hidden.forward(x)
        result = self.output.forward(result)
        return result

    def backward(self, upstream_gradients):
        da_prev = self.output.backward(upstream_gradients)
        self.hidden.backward(da_prev)
        return None
