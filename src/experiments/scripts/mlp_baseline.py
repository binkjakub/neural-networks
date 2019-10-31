import numpy as np
from sklearn.metrics.classification import accuracy_score

from settings import DATA_PATH
from src.data_processing.io import load_mnist
from src.datasets.mnist_dataset import batch_data
from src.nn.losses.mse import MeanSquaredError
from src.nn.networks.mlp import MultiLayerPerceptron
from src.utils.data_utils import to_one_hot

model = MultiLayerPerceptron(784, 10)
loss = MeanSquaredError()

optimizer = None
batch_size = 50
epochs = 20

train_set, val, test = load_mnist(DATA_PATH)

train_set = train_set[0], to_one_hot(train_set[1])
test = test[0], to_one_hot(test[1])


for epoch in range(epochs):
    for x_train, y_train in batch_data(*train_set, batch_size):
        model_output = model.forward(x_train)
        batch_loss = loss.forward(y_train.T, model_output)

        loss_grad = loss.backward(1)
        model.backward(loss_grad)

    model_output = model.forward(val[0])
    model_train = model.forward(train_set[0])
    predictions = np.argmax(model_output, axis=0)

    print(f'EPOCH {epoch}: MSE_TRAIN: {loss.forward(train_set[1].T, model_train)}')
    print(f'EPOCH {epoch}: ACC_TRAIN: '
          f'{accuracy_score(np.argmax(train_set[1], axis=1), np.argmax(model_train, axis=0))}')
    print(f'EPOCH {epoch}: {loss.forward(val[1], model_output)}')
    print(f'EPOCH {epoch}: {accuracy_score(val[1], predictions[:, np.newaxis])}')
    print()
