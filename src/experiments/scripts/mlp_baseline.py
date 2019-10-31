import numpy as np
from sklearn.metrics.classification import accuracy_score

from settings import DATA_PATH
from src.data_processing.io import load_mnist
from src.datasets.mnist_dataset import batch_data
from src.nn.layers.initializers import XAVIER, RANDOM_NORMAL
from src.nn.losses.mse import MeanSquaredError
from src.nn.networks.mlp import MultiLayerPerceptron
from src.utils.data_utils import to_one_hot

model = MultiLayerPerceptron(784, 10, RANDOM_NORMAL)
loss = MeanSquaredError()
batch_size = 64
epochs = 50

train, val, test = load_mnist(DATA_PATH)

train_set = train[0], to_one_hot(train[1])
val_set = val[0], to_one_hot(val[1])
test_set = test[0], to_one_hot(test[1])

for epoch in range(epochs):
    for x_train, y_train in batch_data(*train_set, batch_size):
        model_output = model.forward(x_train.T)
        batch_loss = loss.forward(y_train.T, model_output)

        loss_grad = loss.backward()
        model.backward(loss_grad)

    train_output = model.forward(train_set[0].T)
    val_output = model.forward(val[0].T)
    test_output = model.forward(test[0].T)

    train_pred = np.argmax(train_output, axis=0)
    val_pred = np.argmax(val_output, axis=0)
    test_pred = np.argmax(test_output, axis=0)

    train_accuracy = accuracy_score(train[1], train_pred)
    val_accuracy = accuracy_score(val[1], val_pred)
    test_accuracy = accuracy_score(test[1], test_pred)

    train_loss = loss.forward(train_set[1].T, train_output)
    val_loss = loss.forward(val_set[1].T, val_output)
    test_loss = loss.forward(test_set[1].T, val_output)

    print('-'.ljust(12, '-'), epoch, '-'.ljust(14, '-'))
    print(f'MSE: train: {round(train_loss, 3) :0.3f} '
          f'val: {round(val_loss, 3) :0.3f} '
          f'test {round(test_loss, 3) :0.3f}')
    print(f'ACC: train: {round(train_accuracy, 3) :0.3f} '
          f'test: {round(val_accuracy, 3) :0.3f} '
          f'test: {round(test_accuracy, 3) :0.3f}')
    print()
