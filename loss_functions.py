from abc import ABC, abstractmethod

import numpy as np


def safe_Y_hat(Y_hat, epsilon=1e-15):
    return np.clip(Y_hat, epsilon, 1 - epsilon)


class Loss(ABC):
    def __init__(self, C=0.1, regularization=True) -> None:
        self.C = C
        self.regularization = regularization

    def l2_loss(self, Y_hat, Y, params):
        n = Y_hat.shape[1]
        loss = self._loss(Y_hat, Y)
        param_losses = [np.sum(param**2) for param in params]
        return loss + self.C * (sum(param_losses) / (2 * n))

    def loss(self, Y_hat, Y, params):
        if self.regularization:
            return self.l2_loss(Y_hat, Y, params)
        else:
            return self._loss(Y_hat, Y)

    def dloss_dy_hat(self, Y_hat, Y):
        grad_loss = self._dloss_dy_hat(Y_hat, Y)
        return grad_loss

    def dreg_loss_dparam(self, Y_hat, param):
        n = Y_hat.shape[1]
        return self.C * param / n

    @abstractmethod
    def _loss(self, Y_hat, Y) -> float:
        raise NotImplementedError()

    @abstractmethod
    def _dloss_dy_hat(self, Y_hat, Y) -> np.array:
        raise NotImplementedError()


class BinaryCrossEntropyLoss(Loss):
    def _loss(self, Y_hat, Y) -> float:
        # Y_hat are the predictions
        Y_hat = safe_Y_hat(Y_hat)
        loss = -1 * (Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
        return np.mean(loss)

    def _dloss_dy_hat(self, Y_hat, Y) -> np.array:
        n = Y_hat.shape[1]
        numerator = Y_hat - Y
        return (numerator) / n


class MSE(Loss):
    def _loss(self, Y_hat, Y) -> float:
        # Y_hat are the predictions
        squared_error = (Y_hat - Y) ** 2
        return np.mean(squared_error)

    def _dloss_dy_hat(self, Y_hat, Y) -> np.array:
        n = Y_hat.shape[1]
        return 2 * (Y_hat - Y) / n
