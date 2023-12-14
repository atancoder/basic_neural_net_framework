import numpy as np


def log_loss(Y_hat, Y) -> float:
    # Y_hat are the predictions
    loss = -1 * (Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
    return np.mean(loss)


def dlog_loss_dy_hat(Y_hat, Y) -> np.array:
    numerator = Y_hat - Y
    denom = Y_hat * (1 - Y_hat)
    # Need to figure out dimensions
    return (numerator / denom) / len(Y_hat)
