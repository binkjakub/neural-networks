import numpy as np
from tqdm import tqdm

from settings import MNIST_PATH
from src.data_processing.io import load_mnist
from src.datasets.mnist_dataset import batch_data
from src.metrics.classficiation import accuracy
from src.nn.losses.cross_entropy import CrossEntropyWithLogitLoss
from src.nn.networks.mlp import MultiLayerPerceptron
from src.nn.optimizers.sgd import Momentum
from src.utils.data_utils import to_one_hot

model = MultiLayerPerceptron(input_dim=784,
                             output_dim=10,
                             hidden_sizes=[64],
                             hidden_activation='relu',
                             output_activation='identity',
                             initializer='he',
                             output_initializer='plain')

loss = CrossEntropyWithLogitLoss()
learning_rate = 1e-2
momentum = 0.9
optimizer = Momentum(model.parameters(), learning_rate, momentum)
batch_size = 50
epochs = 100

train, val, test = load_mnist(MNIST_PATH)

train_set = train[0].T, to_one_hot(train[1]).T
val_set = val[0].T, to_one_hot(val[1]).T
test_set = test[0].T, to_one_hot(test[1]).T

for epoch in range(epochs):
    for x_train, y_train in batch_data(*train_set, batch_size):
        model_output = model.forward(x_train)
        batch_loss = loss.forward(y_train, model_output)
        loss_grad = loss.backward()
        model.backward(loss_grad)
        optimizer.step()

    train_output = model.forward(train_set[0])
    val_output = model.forward(val_set[0])
    test_output = model.forward(test_set[0])

    train_pred = np.argmax(train_output, axis=0)
    val_pred = np.argmax(val_output, axis=0)
    test_pred = np.argmax(test_output, axis=0)

    train_accuracy = accuracy(train[1], train_pred)
    val_accuracy = accuracy(val[1], val_pred)
    test_accuracy = accuracy(test[1], test_pred)

    train_loss = loss.forward(train_set[1], train_output)
    val_loss = loss.forward(val_set[1], val_output)
    test_loss = loss.forward(test_set[1], val_output)

    print('-'.ljust(12, '-'), epoch, '-'.ljust(14, '-'))
    print(f'MSE: train: {round(train_loss, 3) :0.3f} '
          f'val: {round(val_loss, 3) :0.3f} '
          f'test {round(test_loss, 3) :0.3f}')
    print(f'ACC: train: {round(train_accuracy, 3) :0.3f} '
          f'val: {round(val_accuracy, 3) :0.3f} '
          f'test: {round(test_accuracy, 3) :0.3f}')
    print()
