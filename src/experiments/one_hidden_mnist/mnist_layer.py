from settings import RESULTS_PATH
from src.data_processing.io import pickle_object
from src.experiments.mnist_experiment import evaluate_mnist_experiment
from src.nn.losses.cross_entropy import CrossEntropyWithLogitLoss
from src.nn.networks.mlp import MultiLayerPerceptron
from src.nn.optimizers.sgd import SGD

"""
Study on layer size influence
Each model is performed multiple times
"""

N_REPEATS = 3
layer_sizes = [10, 32, 128, 512, 1024]
learning_rate = 1e-2
batch_size = 64
epochs = 100


def return_contexts():
    contexts = {}
    for l_size in layer_sizes:
        model = MultiLayerPerceptron(input_dim=784,
                                     output_dim=10,
                                     hidden_sizes=[l_size],
                                     hidden_activation='sigmoid',
                                     output_activation='identity',
                                     initializer='plain')
        loss = CrossEntropyWithLogitLoss()
        optimizer = SGD(model.parameters(), learning_rate)
        contexts[l_size] = (model, loss, optimizer)
    return contexts, batch_size, epochs


results = evaluate_mnist_experiment(return_contexts, N_REPEATS)
pickle_object(results, RESULTS_PATH, 'layer_size_experiment')
