from src.nn.optimizers.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, parameters, learning_rate, ):
        super().__init__(parameters)

        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            param.val = param.val - self.learning_rate * param.grad


class Momentum(SGD):
    def __init__(self, parameters, learning_rate, momentum):
        super().__init__(parameters, learning_rate)
        self.momentum = momentum
        self.past_update = [param.grad for param in self.parameters]

    def step(self):
        for idx, param in enumerate(self.parameters):
            v_param = self.momentum * self.past_update[idx] + self.learning_rate * param.grad
            self.past_update[idx] = v_param
            param.val = param.val - v_param
