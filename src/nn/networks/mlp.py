from src.nn.activations import get_activation
from src.nn.layers.fully_connected import (FullyConnected)
from src.nn.module import Module


class MultiLayerPerceptron(Module):
    def __init__(self, input_dim, output_dim, hidden_sizes, hidden_activation, output_activation,
                 initializer, output_initializer=None):
        super().__init__()
        self.hidden_layers = []

        hidden_inputs = [input_dim] + hidden_sizes
        for i in range(1, len(hidden_inputs)):
            self.hidden_layers.append(FullyConnected(
                in_dim=hidden_inputs[i - 1],
                out_dim=hidden_inputs[i],
                activation=get_activation(hidden_activation),
                initializer=initializer
            ))

        self.output = FullyConnected(in_dim=hidden_sizes[-1],
                                     out_dim=output_dim,
                                     activation=get_activation(output_activation),
                                     initializer=initializer
                                     if output_initializer is None else output_initializer)

    def forward(self, x):
        for h in self.hidden_layers:
            x = h.forward(x)
        x = self.output.forward(x)
        return x

    def backward(self, upstream_gradients):
        upstream_gradients = self.output.backward(upstream_gradients)
        for h in reversed(self.hidden_layers):
            upstream_gradients = h.backward(upstream_gradients)
        return upstream_gradients

    def parameters(self):
        params = []
        for h in self.hidden_layers:
            h_params = h.parameters()
            if h_params is not None:
                params.extend(h_params)
        params.extend(self.output.parameters())
        return params
