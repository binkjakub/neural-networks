import numpy as np

from src.nn.optimizers.optimizer import Optimizer


class AdagradOptimizer(Optimizer):
    def __init__(self, parameters, learning_rate):
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.sum_grads = [np.square(param.grad) for param in self.parameters]

    def step(self):
        for idx, (param, s_grad) in enumerate(zip(self.parameters, self.sum_grads)):
            self.sum_grads[idx] = s_grad + np.square(param.grad)
            update = self.learning_rate / np.sqrt(1e-9 + self.sum_grads[idx]) * param.grad
            param.val = param.val - update
