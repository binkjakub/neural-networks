import pickle


def load_mnist(path):
    with open(path, 'rb') as file:
        training_data, validation_data, test_data = pickle.load(file, encoding='bytes')
    return training_data, validation_data, test_data
