import random
from collections import defaultdict
from typing import List, Union

import numpy as np

from layers import Layer
from loss_functions import dlog_loss_dy_hat, log_loss


# Only supports single class classification
class NeuralNetwork:
    def __init__(self, input_layer_size, layers) -> None:
        self.layers = layers
        prev_layer_size = input_layer_size
        for i in range(len(layers)):
            layer = layers[i]
            if i + 1 < len(layers):
                next_layer = layers[i + 1]
            else:
                next_layer = None

            layer.compile(prev_layer_size, i + 1, next_layer)
            prev_layer_size = layer.units

    def predict(self, input, return_intermediates=False):
        """
        Return intermediates gives us all data that flows through the layers,
        from input all the way to final output
        """
        layer_outputs = defaultdict(dict)
        layer_outputs[0]["output"] = input
        for layer in self.layers:
            input_to_layer = layer_outputs[layer.id - 1]["output"]
            output, z = layer.compute(input_to_layer, include_z=True)
            layer_outputs[layer.id] = {"output": output, "Z": z}

        if return_intermediates:
            return layer_outputs
        else:
            last_layer = self.layers[-1].id
            return layer_outputs[last_layer]["output"]

    def train(self, X, Y, iterations=10):
        for _ in range(iterations):
            self.apply_backprop(X, Y)

    def apply_backprop(self, X, Y):
        layer_outputs = self.predict(X, return_intermediates=True)
        last_layer = self.layers[-1].id
        next_layer_dloss_dinput = dlog_loss_dy_hat(
            Y_hat=layer_outputs[last_layer]["output"], Y=Y
        )
        for layer in reversed(self.layers):
            X_out = layer_outputs[layer.id]["output"]
            Z_out = layer_outputs[layer.id]["Z"]
            X_in = layer_outputs[layer.id - 1]["output"]
            layer.update_params(next_layer_dloss_dinput, X_out, Z_out, X_in)
            next_layer_dloss_dinput = layer.dloss_dinput(next_layer_dloss_dinput, X_out)

    def score(self, X, Y):
        prediction = (self.predict(X) >= 0.5).astype(int)
        num_correct = (Y == prediction).sum()
        print(f"Accuracy: {num_correct / X.shape[1]}")

    def compute_loss(self, Y_hat, Y) -> float:
        # Y_hat are the predictions
        # We only support Log Loss (BinaryCrossEntropy Loss)
        return log_loss(Y_hat, Y)
