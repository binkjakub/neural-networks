import numpy as np


def accuracy(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError(f'Incorrect lengths of data: y_true: {len(y_true)}, y_pred: {len(y_pred)}')
    return np.sum(y_true == y_pred) / len(y_true)
