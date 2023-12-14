import numpy as np


def safe_Y_hat(Y_hat, epsilon=1e-15):
    return np.clip(Y_hat, epsilon, 1 - epsilon)


def log_loss(Y_hat, Y) -> float:
    # Y_hat are the predictions
    Y_hat = safe_Y_hat(Y_hat)
    loss = -1 * (Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
    return np.mean(loss)


def dlog_loss_dy_hat(Y_hat, Y) -> np.array:
    Y_hat = safe_Y_hat(Y_hat)
    numerator = Y_hat - Y
    denom = Y_hat * (1 - Y_hat)
    # Need to figure out dimensions
    return (numerator / denom) / len(Y_hat)
