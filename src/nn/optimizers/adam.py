import numpy as np

from src.nn.optimizers.optimizer import Optimizer


class AdamOptimizer(Optimizer):
    def __init__(self, parameters, learning_rate, beta_1=0.9, beta_2=0.999):
        super().__init__(parameters)
        self.t = 1
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = 1e-7

        self.m_t = [np.zeros(param.grad.shape) for param in self.parameters]
        self.v_t = [np.zeros(param.grad.shape) for param in self.parameters]

        self.prev_m_t = [np.zeros(param.grad.shape) for param in self.parameters]
        self.prev_v_t = [np.zeros(param.grad.shape) for param in self.parameters]

    def step(self):
        state_params = zip(self.parameters, self.prev_m_t, self.prev_v_t)
        for idx, (param, prev_m_t, prev_v_t) in enumerate(state_params):
            m_t = self.beta_1 * prev_m_t + (1 - self.beta_1) * param.grad
            v_t = self.beta_2 * prev_v_t + (1 - self.beta_2) * param.grad ** 2

            self.prev_m_t[idx] = self.m_t[idx]
            self.prev_v_t[idx] = self.v_t[idx]
            self.m_t[idx] = m_t
            self.v_t[idx] = v_t

            m_t_hat = m_t / (1 - np.power(self.beta_1, self.t))
            v_t_hat = v_t / (1 - np.power(self.beta_2, self.t))

            param.val = param.val - (
                    self.learning_rate / (np.sqrt(v_t_hat) + self.epsilon)) * m_t_hat
        self.t += 1
