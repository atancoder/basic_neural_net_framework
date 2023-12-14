import random
from typing import List, Tuple, Union

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(x, 0)


def relu_prime(x):
    return np.where(x > 0, 1, 0)


ACTIVATION_FN_PRIME = {relu: relu_prime, sigmoid: sigmoid_prime}


class Layer:
    def __init__(
        self, units: int, activation_fn: str, learning_rate: float = 1
    ) -> None:
        self.units = units
        self.learning_rate = learning_rate
        if activation_fn == "relu":
            self.activation_fn = relu
        elif activation_fn == "sigmoid":
            self.activation_fn = sigmoid
        else:
            raise Exception(f"{activation_fn} activation fn not supported")
        self._is_compiled = False

        # Below params are set by compilation
        self.input_size = None
        self.W = None  # shape = (# of units, input_size)
        self.b = None  # shape = (# of units, 1)
        self.id = None
        self.next_layer = None

    def compile(self, input_size, id, next_layer):
        self._init_params(input_size)
        self.id = id  # 1st hidden layer is layer id 1
        self.next_layer = next_layer
        self._is_compiled = True

    def apply_activation(self, x):
        # x can be a real number, vector, or matrix of real numbers
        return self.activation_fn(x)

    def compute(
        self, input: np.array, include_z=False
    ) -> Union[np.array, Tuple[np.array, np.array]]:
        """
        input can be a vector or a matrix
        If matrix, each column represents a data sample

        Output is of shape = (# of units, # of input samples)
        """
        if not self._is_compiled:
            raise Exception(
                "Layers are not compiled. Must be used in context of NeuralNetwork class"
            )

        if len(input.shape) == 1:
            num_samples = 1
        else:
            num_samples = input.shape[1]
        b = np.tile(self.b, (1, num_samples))  # Add b to each sample
        WX_b = np.dot(self.W, input) + b
        output = self.apply_activation(WX_b)
        if include_z:
            return output, WX_b
        return output

    def update_params(self, next_layer_dloss_dinput, X_out, Z_out, X_in):
        self.W -= self.learning_rate * self._dloss_dW(
            next_layer_dloss_dinput, X_out, Z_out, X_in
        )
        self.b -= self.learning_rate * self._dloss_db(
            next_layer_dloss_dinput, X_out, Z_out, X_in
        )

    def _dloss_dW(self, next_layer_dloss_dinput, X_out, Z_out, X_in):
        """
        next_layer_dloss_dinput shape = (# of units, # of samples)
        _dX_dZ shape = (# of units, # of samples)
        _dZ_dW shape = (# of samples, input size)

        Return value shape = (# of units, input_size)
        """
        dX_dZ = self._dX_dZ(X_out)
        dZ_dW = self._dZ_dW(X_in)
        matrix_mult = next_layer_dloss_dinput * dX_dZ
        return np.dot(matrix_mult, dZ_dW)

    def _dloss_db(self, next_layer_dloss_dinput, X_out, Z_out, X_in):
        """
        next_layer_dloss_dinput shape = (# of units, # of samples)
        _dX_dZ shape = (# of units, # of samples)
        _dZ_dB shape = (# of units, # of samples)

        Return value shape = (# of units, 1)
        """
        dX_dZ = self._dX_dZ(X_out)
        dZ_dB = self._dZ_dB(Z_out)
        dloss_dB = next_layer_dloss_dinput * dX_dZ * dZ_dB

        # Convert to b.shape by taking average across samples (along the column)
        return np.mean(dloss_dB, axis=1).reshape(self.b.shape)

    def _dX_dZ(self, X_out) -> List:
        """
        X_out shape = (# of units, # of samples)
        Returns X_out shape
        """
        return ACTIVATION_FN_PRIME[self.activation_fn](X_out)

    def _dZ_dW(self, X_in) -> List:
        """
        X_in shape = (# of prev_layer units, # of samples)

        note: input size == # of prev_layer units
        Returns (# of samples, input size)
        """
        return X_in.T

    def _dZ_dB(self, Z_out):
        """
        Z_out shape = (# of units, # of samples)

        Return value shape = Z_out shape
        """
        return np.ones(Z_out.shape)

    def dloss_dinput(self, next_layer_dloss_dinput, X_out) -> float:
        """
        We expose this function in each layer as it's used to backprop to the prev layer

        next_layer_dloss_dinput shape = (# of units, # of samples)
        dX_dZ shape = (# of units, # of samples)
        dZ_dinput shape = (# of units, input_size)

        Return value shape =  (input_size, # of samples)
        """
        dX_dZ = self._dX_dZ(X_out)
        dZ_dinput = self._dZ_dinput()
        return np.dot(dZ_dinput.T, next_layer_dloss_dinput * dX_dZ)

    def _dZ_dinput(self):
        """
        This is just self.W
        Return value shape = (# of units, input_size)
        """
        return self.W

    def _init_params(self, input_size) -> None:
        self.input_size = input_size
        weight_vectors = []
        bias_terms = []
        for _ in range(self.units):
            weight_vectors.append(np.random.rand(self.input_size))
            bias_terms.append(random.random())
        weight_vectors = np.array(weight_vectors)
        bias_terms = np.array(bias_terms)
        self.W = weight_vectors
        self.b = bias_terms.reshape(-1, 1)
