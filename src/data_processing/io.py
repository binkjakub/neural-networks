import os
import pickle as pkl


def load_mnist(path):
    with open(path, 'rb') as file:
        training_data, validation_data, test_data = pkl.load(file, encoding='bytes')
    return training_data, validation_data, test_data


def pickle_object(obj, path, filename):
    filename = filename if filename.endswith('.pkl') else filename + '.pkl'
    full_path = os.path.join(path, filename)
    with open(full_path, 'wb') as file:
        pkl.dump(obj, file, protocol=pkl.HIGHEST_PROTOCOL)
