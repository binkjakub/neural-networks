from abc import ABC, abstractmethod

from src.nn.layers.initializers import Parameter


class Optimizer(ABC):
    def __init__(self, parameters):
        assert all(isinstance(param, Parameter) for param in parameters), 'Invalid parameters types'
        self.parameters = parameters

    @abstractmethod
    def step(self):
        return
