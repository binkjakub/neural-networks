from src.datasets.mnist_dataset import batch_data


def train_mnist(model, loss, optimizer, epochs):
    for epoch in range(epochs):
        for x_train, y_train in batch_data(*train_set, batch_size):
            model_output = model.forward(x_train.T)
            batch_loss = loss.forward(y_train.T, model_output)

            loss_grad = loss.backward()
            model.backward(loss_grad)
            optimizer.step()
