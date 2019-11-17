import numpy as np

from src.nn.optimizers.optimizer import Optimizer


class AdadeltaOptimizer(Optimizer):
    def __init__(self, parameters, decay_rate=0.99):
        super().__init__(parameters)
        self.epsilon = 1e-6
        self.decay_rate = decay_rate

        self.prev_grad = [np.zeros(shape=param.grad.shape) for param in self.parameters]
        self.prev_update = [np.zeros(shape=param.grad.shape) for param in self.parameters]

    def step(self):
        for idx, param in enumerate(self.parameters):
            accum_grad = self.decay_rate * self.prev_grad[idx] + (
                    1 - self.decay_rate) * param.grad ** 2

            rms_grad_t = np.sqrt(accum_grad + self.epsilon)
            rms_prev_update = np.sqrt(self.prev_update[idx]+self.epsilon)

            update = rms_prev_update * param.grad / rms_grad_t

            self.prev_update[idx] = self.decay_rate * self.prev_update[idx] + (
                    1 - self.decay_rate) * update ** 2
            self.prev_grad[idx] = accum_grad

            param.val = param.val - update
