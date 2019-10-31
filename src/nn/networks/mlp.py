from src.nn.activations.sigmoid import Sigmoid
# from src.nn.activations.softmax import Softmax
from src.nn.layers.fully_connected import (FullyConnected)
from src.nn.module import Module


class MultiLayerPerceptron(Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.hidden = FullyConnected(in_shape=input_shape,
                                     out_shape=256,
                                     activation=Sigmoid())

        self.output = FullyConnected(in_shape=256,
                                     out_shape=output_shape,
                                     activation=Sigmoid())

    def forward(self, x):
        result = self.hidden.forward(x.T)
        result = self.output.forward(result)
        return result

    def backward(self, upstream_gradients):
        out_grad = self.output.backward(upstream_gradients)
        out_grad = self.hidden.backward(out_grad)
        return out_grad
