import random

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Layer:
    def __init__(self, units: int, activation_fn: str) -> None:
        self.units = units
        self.activation_fn = activation_fn
        self.input_size = None
        self.W = None
        self.B = None
        self.layer_id = None

    def apply_activation(self, x):
        # x can be a real number of vector of real numbers
        if self.activation_fn == "relu":
            return np.maximum(x, 0)
        if self.activation_fn == "sigmoid":
            return sigmoid(x)

    def forward_prop(self, input: np.array) -> np.array:
        Wx_B = np.dot(self.W, input) + self.B
        output = self.apply_activation(Wx_B)

        print(f"Layer {self.layer_id}: Transform {input} --> {output}")
        return output

    def set_input_size(self, input_size) -> None:
        self.input_size = input_size
        weight_vectors = []
        bias_terms = []
        for _ in range(self.units):
            weight_vectors.append(np.random.rand(self.input_size))
            bias_terms.append(random.random())
        weight_vectors = np.array(weight_vectors)
        bias_terms = np.array(bias_terms)
        self.W = weight_vectors
        self.B = bias_terms

    def set_layer_id(self, i: int) -> None:
        self.layer_id = i


class NeuralNetwork:
    def __init__(self, input_layer_size, layers) -> None:
        self.layers = layers
        prev_layer_size = input_layer_size
        for i, layer in enumerate(layers):
            layer.set_layer_id(i)
            layer.set_input_size(prev_layer_size)
            prev_layer_size = layer.units

    def forward_prop(self, input) -> float:
        for layer in self.layers:
            input = layer.forward_prop(input)

        return input


input = np.array([2, 4])
layers = [Layer(3, "relu"), Layer(2, "relu"), Layer(1, "sigmoid")]

nn = NeuralNetwork(input.size, layers)
print(nn.forward_prop(input))
