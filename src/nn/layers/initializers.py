import numpy as np

RANDOM_NORMAL = 'plain'
UNIFORM = 'uniform'
XAVIER = 'xavier'
HE = 'he'

RAND_N_VAR = 0.1


def init_parameters(in_dim, out_dim, initializer=RANDOM_NORMAL):
    bias = np.zeros(shape=[out_dim, 1])
    if initializer == XAVIER:
        weights = np.random.rand(out_dim, in_dim) / np.sqrt(in_dim)
    elif initializer == HE:
        weights = np.random.randn(out_dim, in_dim) * np.sqrt(1 / in_dim)
    elif initializer == UNIFORM:
        weights = np.random.uniform(-np.sqrt(in_dim), -np.sqrt(out_dim), size=[out_dim, in_dim])
    else:
        weights = np.random.normal(loc=0., scale=RAND_N_VAR, size=[out_dim, in_dim])
    return weights, bias
