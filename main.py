import random

import numpy as np
from sklearn.datasets import make_moons

from layers import Layer
from loss_functions import MSE, BinaryCrossEntropyLoss
from neural_net import NeuralNetwork
from tensor_flow import relu_neural_net


def in_range(x, y) -> bool:
    return y < 0.3 * x + 2


def generate_train_data_classification(samples=10000):
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


def generate_train_data_regression(samples=10000):
    data = []
    labels = []
    for _ in range(samples):
        x1 = random.uniform(0, 100)
        x2 = random.uniform(0, 100)
        data.append((x1, x2))
        labels.append(3 * x1 - 4 * x2)
    return np.array(data).T, np.array(labels).reshape(1, -1)


def regression():
    X, Y = generate_train_data_regression(samples=1000)
    X_test, Y_test = generate_train_data_regression(samples=100)
    # layers = [Layer(3, "relu"), Layer(2, "relu"), Layer(1, "sigmoid")]
    # layers = [[Layer(2, "relu"),] Layer(1, "sigmoid")]
    layers = [Layer(1, "linear")]
    nn = NeuralNetwork(
        2,
        layers,
        loss_cls=MSE(C=1, regularization=False),
        dynamic_lr=True,
        min_lr=1e-15,
    )

    Y_hat = nn.predict(X_test)
    loss = nn.compute_loss(Y_hat, Y_test)
    print(f"Loss before training: {loss}")

    nn.train(X, Y)
    nn.print_weights()
    Y_hat = nn.predict(X_test)
    loss = nn.compute_loss(Y_hat, Y_test)
    print(f"Loss after training: {loss}")


def tensor_flow():
    X, Y = generate_train_data_classification(samples=10000)
    relu_neural_net(X, Y)


def classification():
    X, Y = generate_train_data_classification(samples=10000)
    # layers = [Layer(3, "relu"), Layer(2, "relu"), Layer(1, "sigmoid")]
    # layers = [Layer(2, "relu"), Layer(1, "sigmoid")]
    layers = [Layer(1, "sigmoid")]
    nn = NeuralNetwork(
        2,
        layers,
        loss_cls=BinaryCrossEntropyLoss(C=0.01, regularization=False),
        learning_rate=0.01,
        dynamic_lr=False,
    )
    nn.score(X, Y)
    Y_hat = nn.predict(X)
    loss = nn.compute_loss(Y_hat, Y)
    print(f"Loss before training: {loss}")

    nn.train(X, Y, iterations=1000)
    nn.print_weights()
    nn.score(X, Y)
    Y_hat = nn.predict(X)
    loss = nn.compute_loss(Y_hat, Y)
    print(f"Loss after training: {loss}")


def main():
    tensor_flow()


if __name__ == "__main__":
    main()
