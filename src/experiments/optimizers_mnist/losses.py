"""
Study on batch size influence
Each model is performed multiple times
"""
from copy import deepcopy

from settings import RESULTS_PATH
from src.data_processing.io import pickle_object
from src.experiments.mnist_experiment import mnist_experiment_on_models
from src.nn.losses.cross_entropy import CrossEntropyWithLogitLoss
from src.nn.losses.mse import MeanSquaredError
from src.nn.networks.mlp import MultiLayerPerceptron
from src.nn.optimizers.sgd import Momentum

N_REPEATS = 3
layer_size = 64
batch_size = 64
epochs = 60


def return_contexts():
    contexts = {}
    mse_model = MultiLayerPerceptron(input_dim=784,
                                     output_dim=10,
                                     hidden_sizes=[layer_size],
                                     hidden_activation='sigmoid',
                                     output_activation='softmax',
                                     initializer='xavier')

    mse_loss = MeanSquaredError()
    mse_optim = Momentum(mse_model.parameters(), learning_rate=1e-3, momentum=0.9)
    contexts['MSE'] = (mse_model, mse_loss, mse_optim)

    ce_model = deepcopy(mse_model)
    ce_model.change_output_activation('identity')
    ce_loss = CrossEntropyWithLogitLoss()
    ce_optim = Momentum(ce_model.parameters(), learning_rate=1e-3, momentum=0.9)
    contexts['Cross Entropy'] = (ce_model, ce_loss, ce_optim)

    return contexts, batch_size, epochs


results = mnist_experiment_on_models(return_contexts, N_REPEATS)

pickle_object(results, RESULTS_PATH, 'losses_result')
