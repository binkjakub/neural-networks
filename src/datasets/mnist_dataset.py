import numpy as np


def batch_data(x, y, batch_size):
    assert x.shape[1] == y.shape[1]
    x, y = _shuffle(x, y)

    current_batch = 0
    batch_end = 0

    while batch_end < x.shape[1]:
        batch_start = current_batch * batch_size
        batch_end = min(batch_start + batch_size, x.shape[1])
        current_batch += 1
        yield x[:, batch_start:batch_end], y[:, batch_start:batch_end]


def _shuffle(x, y):
    shuffled_mask = np.arange(0, x.shape[1])
    np.random.shuffle(shuffled_mask)
    x = x[:, shuffled_mask]
    y = y[:, shuffled_mask]
    return x, y
