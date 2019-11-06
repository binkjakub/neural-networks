"""
Study on batch size influence
Each model is performed multiple times
"""
from settings import RESULTS_PATH
from src.data_processing.io import pickle_object
from src.experiments.mnist_experiment import evaluate_mnist_experiment
from src.nn.losses.cross_entropy import CrossEntropyWithLogitLoss
from src.nn.networks.mlp import MultiLayerPerceptron
from src.nn.optimizers.sgd import SGD

N_REPEATS = 10
layer_size = 128
learning_rate = 1e-2
batch_size = 64
epochs = 50
initializers = ['plain_0.01', 'plain_0.1', 'plain_1', 'xavier']


def return_contexts():
    contexts = {}
    for init in initializers:
        model = MultiLayerPerceptron(input_dim=784,
                                     output_dim=10,
                                     hidden_sizes=[layer_size],
                                     hidden_activation='sigmoid',
                                     output_activation='identity',
                                     initializer=int)
        loss = CrossEntropyWithLogitLoss()
        optimizer = SGD(model.parameters(), learning_rate)
        contexts[init] = (model, loss, optimizer)
    return contexts


results = evaluate_mnist_experiment(return_contexts, N_REPEATS)

pickle_object(results, RESULTS_PATH, 'sigmoid_relu_experiment')
