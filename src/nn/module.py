from abc import ABC, abstractmethod


class Module(ABC):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        return

    @abstractmethod
    def backward(self, upstream_gradients):
        return

    @abstractmethod
    def parameters(self):
        return
