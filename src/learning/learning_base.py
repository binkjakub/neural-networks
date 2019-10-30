

class Learning:
    def __init__(self, model, loss, optimizer, learning_rate):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def forward(self, x):
        pass

    def backward(self):
        pass
