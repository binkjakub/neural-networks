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
        self.last_grads = [param.grad for param in self.parameters]

    def step(self):
        for param, last_grad in zip(self.parameters, self.last_grads):
            v_param = self.momentum * last_grad + self.learning_rate * param.grad
            param.val = param.val - v_param
