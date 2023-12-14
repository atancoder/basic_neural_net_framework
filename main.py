import random

import numpy as np

from layers import Layer
from neural_net import NeuralNetwork


def in_range(x, y) -> bool:
    return y < 0.3 * x + 2


def generate_train_data(samples=10000):
    data = []
    labels = []
    for _ in range(samples):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        data.append((x, y))
        if in_range(x, y):
            labels.append(1)
        else:
            labels.append(0)
    return np.array(data).T, np.array(labels).reshape(1, -1)


def main():
    X, Y = generate_train_data(samples=10000)
    X_test, Y_test = generate_train_data(samples=100)
    # layers = [Layer(3, "relu"), Layer(2, "relu"), Layer(1, "sigmoid")]
    # layers = [[Layer(2, "relu"),] Layer(1, "sigmoid")]
    layers = [Layer(1, "sigmoid")]
    nn = NeuralNetwork(2, layers)

    nn.score(X_test, Y_test)
    Y_hat = nn.predict(X_test)
    loss = nn.compute_loss(Y_hat, Y_test)
    print(f"Loss before training: {loss}")

    nn.train(X, Y)
    nn.print_weights()
    nn.score(X_test, Y_test)
    Y_hat = nn.predict(X_test)
    loss = nn.compute_loss(Y_hat, Y_test)
    print(f"Loss after training: {loss}")


if __name__ == "__main__":
    main()
