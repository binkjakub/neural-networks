import numpy as np

from settings import DATA_PATH
from src.data_processing.io import load_mnist
from src.datasets.mnist_dataset import batch_data
from src.layers.losses import MeanSquaredError
from src.models.mlp import MultiLayerPerceptron
from src.utils.data_utils import to_one_hot
from sklearn.metrics.classification import accuracy_score
model = MultiLayerPerceptron(784, 10)
loss = MeanSquaredError()

optimizer = None
batch_size = 50
epochs = 10

train_set, val, test = load_mnist(DATA_PATH)
train_set = train_set[0], to_one_hot(train_set[1])
# val = val[0], to_one_hot(val[1])
test = test[0], to_one_hot(test[1])

for epoch in range(epochs):
    for x_train, y_train in batch_data(*train_set, batch_size):
        model_output = model.forward(x_train)
        batch_loss = loss.forward(y_train, model_output)

        loss_grad = loss.backward(1)
        model.backward(loss_grad)

    model_output = model.forward(val[0])
    predictions = np.argmax(model_output, axis=1)
    print(f'EPOCH {epoch}: {accuracy_score(val[1], predictions[:, np.newaxis])}')

