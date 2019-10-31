import numpy as np

RANDOM_NORMAL = 'plain'
UNIFORM = 'uniform'
XAVIER = 'xavier'
HE = 'he'

RAND_N_VAR = 0.1


class Parameter:
    def __init__(self, val):
        self.val = val
        self.grad = np.zeros(shape=self.val.shape)


def init_parameters(in_dim, out_dim, initializer=RANDOM_NORMAL):
    bias = Parameter(np.zeros(shape=[out_dim, 1]))
    if initializer == XAVIER:
        weights = Parameter(
            np.random.normal(scale=np.sqrt(2. / (in_dim + out_dim)), size=[out_dim, in_dim]))
    elif initializer == HE:
        weights = Parameter(np.random.normal(scale=np.sqrt(in_dim), size=[out_dim, in_dim]))
    elif initializer == UNIFORM:
        weights = Parameter(
            np.random.uniform(-np.sqrt(in_dim), -np.sqrt(out_dim), size=[out_dim, in_dim]))
    else:
        weights = Parameter(np.random.normal(loc=0., scale=RAND_N_VAR, size=[out_dim, in_dim]))
    return weights, bias
