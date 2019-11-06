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
layer_size = 64
learning_rate = 1e-2
batch_size = 64
epochs = 100
activations = ['sigmoid', 'relu']


def return_contexts():
    contexts = {}
    base_activation = activations[0]
    base_model = MultiLayerPerceptron(input_dim=784,
                                      output_dim=10,
                                      hidden_sizes=[layer_size],
                                      hidden_activation=base_activation,
                                      output_activation='identity',
                                      initializer='plain')
    base_loss = CrossEntropyWithLogitLoss()
    base_optimizer = SGD(base_model.parameters(), learning_rate)
    contexts[base_activation] = (base_model, base_loss, base_optimizer)

    for act in activations[1:]:
        model = deepcopy(base_model)
        model.change_hidden_activation(act)
        loss = CrossEntropyWithLogitLoss()
        optimizer = SGD(model.parameters(), learning_rate)
        contexts[act] = (model, loss, optimizer)

    return contexts, batch_size, epochs


results = evaluate_mnist_experiment(return_contexts, N_REPEATS)

pickle_object(results, RESULTS_PATH, 'sigmoid_relu_experiment')
