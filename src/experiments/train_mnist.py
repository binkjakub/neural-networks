import numpy as np

from settings import MNIST_PATH
from src.data_processing.io import load_mnist
from src.datasets.mnist_dataset import batch_data
from src.metrics.classficiation import accuracy
from src.utils.data_utils import to_one_hot

train, val, test = load_mnist(MNIST_PATH)
train_set = train[0].T, to_one_hot(train[1]).T
val_set = val[0].T, to_one_hot(val[1]).T
test_set = test[0].T, to_one_hot(test[1]).T


def train_mnist(learning_context, epochs=100, batch_size=64, repeat_context=0):
    learning_logs = []

    # pre evaluation
    evaluate_multiple_models(learning_context, 0, repeat_context, learning_logs)

    for epoch in range(1, epochs + 1):
        for x_train, y_train in batch_data(*train_set, batch_size):
            for name, context in learning_context.items():
                model, loss, optimizer = context
                model_output = model.forward(x_train)
                loss.forward(y_train, model_output)

                loss_grad = loss.backward()
                model.backward(loss_grad)
                optimizer.step()

        # log epoch
        evaluate_multiple_models(learning_context, epoch, repeat_context, learning_logs)

    return learning_logs


def evaluate_multiple_models(learning_context, epoch, repeat_context, logs):
    for name, context in learning_context.items():
        model, loss, optimizer = context
        metrics = evaluate_metrics(model, loss)
        metrics.update({'repeat': repeat_context, 'name': name, 'epoch': epoch})
        logs.append(metrics)


def evaluate_metrics(model, loss):
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
    return {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss,
            'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy}
