from tqdm import trange, tqdm

from src.experiments.train_mnist import train_mnist


def evaluate_mnist_experiment(context_getter, n_repeats):
    results = []
    for i in trange(n_repeats):
        for context in context_getter():
            learning_context, batch_size, epochs = context
            results.append(train_mnist(learning_context,
                                       batch_size=batch_size,
                                       epochs=epochs,
                                       repeat_context=i))
    return results


def mnist_experiment_on_models(context_getter, n_repeats):
    results = []
    for i in tqdm(range(n_repeats)):
        learning_context, batch_size, epochs = context_getter()
        results.append(train_mnist(learning_context,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   repeat_context=i))
    return results
