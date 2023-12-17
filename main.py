import random

import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from layers import Layer
from loss_functions import MSE, BinaryCrossEntropyLoss
from neural_net import NeuralNetwork
from py_torch import neural_net as pt_neural_net
from tensor_flow import neural_net as tf_neural_net

np.set_printoptions(threshold=10)


def in_range(x, y) -> bool:
    return y < 0.3 * x + 2


def generate_train_data_classification(samples=10000):
    data = []
    labels = []
    for _ in range(samples):
        x1 = random.uniform(0, 100)
        x2 = random.uniform(0, 100)
        data.append((x1, x2))
        if in_range(x1, x2):
            labels.append(1)
        else:
            labels.append(0)
    return np.array(data), np.array(labels).reshape(-1, 1)


def generate_train_data_regression(samples=10000):
    data = []
    labels = []
    for _ in range(samples):
        x1 = random.uniform(0, 100)
        x2 = random.uniform(0, 100)
        data.append((x1, x2))
        labels.append(3 * x1 - 4 * x2)
    return np.array(data), np.array(labels)


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


def tensor_flow(X, Y, iterations):
    tf_neural_net(X, Y, iterations)


def py_torch(X, Y, iterations):
    pt_neural_net(X, Y, iterations)


def classification(X, Y, iterations):
    X = X.T
    Y = Y.reshape(1, -1)
    layers = [
        Layer(16, "relu"),
        Layer(8, "relu"),
        Layer(4, "relu"),
        Layer(2, "relu"),
        Layer(1, "sigmoid"),
    ]
    nn = NeuralNetwork(
        2,
        layers,
        loss_cls=BinaryCrossEntropyLoss(C=0.01, regularization=False),
        learning_rate=0.1,
        dynamic_lr=True,
    )
    nn.score(X, Y)
    Y_hat = nn.predict(X)
    loss = nn.compute_loss(Y_hat, Y)
    print(f"Loss before training: {loss}")

    nn.train(X, Y, iterations)
    # nn.print_weights()
    nn.score(X, Y)
    Y_hat = nn.predict(X)
    loss = nn.compute_loss(Y_hat, Y)
    print(f"Loss after training: {loss}")


def normalize(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def main():
    X, Y = generate_train_data_classification(samples=10000)
    X = normalize(X)
    classification(X, Y, 1000)


if __name__ == "__main__":
    main()
