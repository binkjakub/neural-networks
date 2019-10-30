import numpy as np


def to_one_hot(data):
    max_label = np.max(data)
    one_hot_vectors = np.zeros(shape=[len(data), max_label + 1])
    one_hot_vectors[np.arange(len(data)), data] = 1
    return one_hot_vectors
