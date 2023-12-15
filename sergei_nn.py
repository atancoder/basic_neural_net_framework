import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons


class ANN:
    """
    Artificial Neural Network class.
    It's uses xavier initialization, ReLU as non-linearity for all layers except
    the last (it uses sigmoid) and a cross-entropy loss function.
    """

    def __init__(self, dims, lambd=0, reg_type=0):
        """
        dims -- number of nodes for each layer.
        lambd -- lambda coefficient for L2 regularization.
        reg_type -- regularization type:
          0 -- L2 regularization cost without 1/m factor;
          1 -- with 1/m factor.
        """
        np.random.seed(2)
        self.params = dict()
        self.L = len(dims) - 1
        self.m = len(Y)
        self.lambd = lambd
        self.reg_type = reg_type
        for i in range(1, self.L + 1):
            self.params[f"W{i}"] = np.random.randn(dims[i], dims[i - 1]) / np.sqrt(
                dims[i - 1]
            )
            self.params[f"b{i}"] = np.zeros((dims[i], 1))

    def forward(self, X, Y=None):
        """
        Forward propogation.
        """
        self.X = X
        self.Y = Y
        self.cache = dict()
        self.cache["A0"] = X
        for i in range(1, self.L + 1):
            self.cache[f"Z{i}"] = (
                np.dot(self.params[f"W{i}"], self.cache[f"A{i-1}"])
                + self.params[f"b{i}"]
            )
            if i == self.L:
                self.cache[f"A{i}"] = 1 / (1 + np.exp(-self.cache[f"Z{i}"]))
            else:
                self.cache[f"A{i}"] = np.maximum(self.cache[f"Z{i}"], 0)
        if Y is not None:
            self.J = (
                -1
                / self.m
                * np.sum(
                    np.log(self.cache[f"A{self.L}"]) * Y
                    + np.log(1 - self.cache[f"A{self.L}"]) * (1 - Y)
                )
            )
            if self.reg_type == 0:
                for i in range(1, self.L + 1):
                    self.J += self.lambd / 2 * np.sum(self.params[f"W{i}"] ** 2)
            else:
                for i in range(1, self.L + 1):
                    self.J += (
                        1 / self.m * self.lambd / 2 * np.sum(self.params[f"W{i}"] ** 2)
                    )

    def backward(self):
        """
        Backward propogation.
        """
        for i in reversed(range(1, self.L + 1)):
            if i == self.L:
                self.cache[f"dZ{i}"] = self.cache[f"A{i}"] - self.Y
            else:
                self.cache[f"dA{i}"] = np.dot(
                    self.params[f"W{i+1}"].T, self.cache[f"dZ{i+1}"]
                )
                self.cache[f"dZ{i}"] = np.multiply(
                    self.cache[f"dA{i}"], np.int64(self.cache[f"A{i}"] > 0)
                )
            self.cache[f"dW{i}"] = (
                1 / self.m * np.dot(self.cache[f"dZ{i}"], self.cache[f"A{i-1}"].T)
            )
            if self.reg_type == 0:
                self.cache[f"dW{i}"] += self.lambd * self.params[f"W{i}"]
            else:
                self.cache[f"dW{i}"] += self.lambd / self.m * self.params[f"W{i}"]
            self.cache[f"db{i}"] = (
                1 / self.m * np.sum(self.cache[f"dZ{i}"], axis=1, keepdims=True)
            )

    def predict(self, X):
        """
        Make predictions.
        """
        self.forward(X)
        return self.cache[f"A{self.L}"] > 0.5

    def update(self, alpha):
        """
        Update weigths and biases.
        alpha -- leaning rate.
        """
        for key, _ in self.params.items():
            self.params[key] -= self.cache[f"d{key}"] * alpha


X, Y = make_moons(n_samples=30, noise=0.3, random_state=1)
X = X.T

dims = [2, 1]
nn = ANN(dims, lambd=0)
for epoch in range(30000 + 1):
    nn.forward(X, Y)
    nn.backward()
    nn.update(0.005)
    if epoch % 3000 == 0:
        print(f"Epoch {epoch}: {nn.J}")

Y_hat = nn.predict(X)
num_correct = (Y == Y_hat).sum()
print(f"Accuracy: {num_correct / Y.shape[0]}")
print(f"Weights: {nn.params}")
