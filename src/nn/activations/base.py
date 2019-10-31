from abc import ABC

from src.nn.module import Module


class Activation(Module, ABC):
    def __init__(self):
        super().__init__()
        self.activation = None
        self.dZ = None
