"""
Study on batch size influence
Each model is performed multiple times
"""
from copy import deepcopy

from settings import RESULTS_PATH
from src.data_processing.io import pickle_object
from src.experiments.mnist_experiment import evaluate_mnist_experiment
from src.nn.losses.cross_entropy import CrossEntropyWithLogitLoss
from src.nn.networks.mlp import MultiLayerPerceptron
from src.nn.optimizers.sgd import SGD

N_REPEATS = 3
batch_sizes = [1, 10, 32, 64, 128, 256, 512, 1024]
learning_rate = 1e-2
layer_size = 32
epochs = 20


def return_contexts():
    multiple_contexts = []
    base_batch_size = batch_sizes[0]
    base_model = MultiLayerPerceptron(input_dim=784,
                                      output_dim=10,
                                      hidden_sizes=[layer_size],
                                      hidden_activation='sigmoid',
                                      output_activation='identity',
                                      initializer='plain')
    base_loss = CrossEntropyWithLogitLoss()
    base_optimizer = SGD(base_model.parameters(), learning_rate)
    contexts = {}
    contexts[base_batch_size] = (base_model, base_loss, base_optimizer)
    multiple_contexts.append((contexts, base_batch_size, epochs))

    for b_size in batch_sizes[1:]:
        contexts = {}
        model = deepcopy(base_model)
        loss = CrossEntropyWithLogitLoss()
        optimizer = SGD(model.parameters(), learning_rate)

        contexts[b_size] = (model, loss, optimizer)
        multiple_contexts.append((contexts, b_size, epochs))

    return multiple_contexts


results = evaluate_mnist_experiment(return_contexts, N_REPEATS)

pickle_object(results, RESULTS_PATH, 'batch_size_experiment')
