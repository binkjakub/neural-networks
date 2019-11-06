"""
Study on batch size influence
Each model is performed multiple times
"""
from copy import deepcopy

from settings import RESULTS_PATH
from src.data_processing.io import pickle_object
from src.experiments.mnist_experiment import evaluate_mnist_experiment, mnist_experiment_on_models
from src.nn.losses.cross_entropy import CrossEntropyWithLogitLoss
from src.nn.networks.mlp import MultiLayerPerceptron
from src.nn.optimizers.sgd import SGD

N_REPEATS = 3
layer_size = 64
learning_rate = 1e-2
batch_size = 64
epochs = 30
initializers = ['plain', 'xavier', 'he']
# initializers = ['plain', 'xavier']


def return_contexts():
    contexts = {}
    for init in initializers:
        out_init = None if init != 'he' else 'plain'
        model = MultiLayerPerceptron(input_dim=784,
                                     output_dim=10,
                                     hidden_sizes=[layer_size],
                                     hidden_activation='relu',
                                     output_activation='identity',
                                     initializer=init,
                                     output_initializer=out_init)
        loss = CrossEntropyWithLogitLoss()
        optimizer = SGD(model.parameters(), learning_rate)
        contexts[init] = (model, loss, optimizer)
    return contexts, batch_size, epochs


results = mnist_experiment_on_models(return_contexts, N_REPEATS)

pickle_object(results, RESULTS_PATH, 'initialization_plain_xavier_experiment')
