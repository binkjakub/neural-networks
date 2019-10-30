import numpy as np


def batch_data(x, y, batch_size):
    assert len(x) == len(y)
    x, y = _shuffle(x, y)

    current_batch = 0
    batch_end = 0

    while batch_end < len(x):
        batch_start = current_batch * batch_size
        batch_end = min(batch_start + batch_size, len(x))
        yield x[batch_start:batch_end], y[batch_start:batch_end]


def _shuffle(x, y):
    shuffled_mask = np.arange(0, len(x))
    np.random.shuffle(shuffled_mask)
    x = x[shuffled_mask]
    y = y[shuffled_mask]
    return x, y
