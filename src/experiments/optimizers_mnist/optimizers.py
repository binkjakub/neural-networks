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
from src.nn.optimizers.adadelta import AdadeltaOptimizer
from src.nn.optimizers.adagrad import AdagradOptimizer
from src.nn.optimizers.adam import AdamOptimizer
from src.nn.optimizers.sgd import SGD, Momentum

N_REPEATS = 3
layer_size = 64
batch_size = 64
epochs = 60
optimizers = {
    'SGD': (SGD, {'learning_rate': 1e-3}),
    'Momentum': (Momentum, {'learning_rate': 1e-3, 'momentum': 0.9}),
    'Adagrad': (AdagradOptimizer, {'learning_rate': 1e-3}),
    'Adadelta': (AdadeltaOptimizer, {'decay_rate': 0.95}),
    'Adam': (AdamOptimizer, {'learning_rate': 1e-3, 'beta_1': 0.9, 'beta_2': 0.999})
}


def return_contexts():
    contexts = {}
    base_model = MultiLayerPerceptron(input_dim=784,
                                      output_dim=10,
                                      hidden_sizes=[layer_size],
                                      hidden_activation='sigmoid',
                                      output_activation='identity',
                                      initializer='xavier')

    for optim_name, optim in optimizers.items():
        model = deepcopy(base_model)
        loss = CrossEntropyWithLogitLoss()
        optimizer = optim[0](model.parameters(), **optim[1])
        contexts[optim_name] = (model, loss, optimizer)

    return contexts, batch_size, epochs


results = mnist_experiment_on_models(return_contexts, N_REPEATS)

pickle_object(results, RESULTS_PATH, 'optimizers_result')
