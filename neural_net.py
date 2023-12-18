import random
from collections import defaultdict
from typing import List, Union

import numpy as np

from layers import Layer
from loss_functions import Loss


# Only supports single class classification
class NeuralNetwork:
    def __init__(
        self,
        input_layer_size,
        layers,
        loss_cls: Loss,
        learning_rate: float = 1,
        dynamic_lr=True,
        min_lr=1e-10,
        lr_epsilon=1e-2,
    ) -> None:
        self.layers = layers
        prev_layer_size = input_layer_size
        self.loss_cls = loss_cls
        self.learning_rate = learning_rate
        self.dynamic_lr = dynamic_lr
        self.min_lr = min_lr
        self.lr_epsilon = lr_epsilon
        self.curr_loss = np.inf
        self.layer_outputs = defaultdict(dict)
        for i in range(len(layers)):
            layer = layers[i]
            if i + 1 < len(layers):
                next_layer = layers[i + 1]
            else:
                next_layer = None

            layer.compile(prev_layer_size, i + 1, next_layer)
            prev_layer_size = layer.units

    def predict(self, input):
        self.layer_outputs = defaultdict(dict)
        self.layer_outputs[0]["output"] = input
        for layer in self.layers:
            input_to_layer = self.layer_outputs[layer.id - 1]["output"]
            output, z = layer.compute(input_to_layer, include_z=True)
            self.layer_outputs[layer.id] = {"output": output, "Z": z}

        last_layer = self.layers[-1].id
        return self.layer_outputs[last_layer]["output"]

    def print_weights(self):
        print("Printing Weights")
        for layer in self.layers:
            print(f"Layer {layer.id}: W = {layer.W}. b = {layer.b}")

    def get_all_params(self):
        params = []
        for layer in self.layers:
            params += [layer.W, layer.b]
        return params

    def adjust_learning_rate(self, new_loss):
        if new_loss > self.curr_loss:
            new_lr = max(self.learning_rate / 2, self.min_lr)
            if new_lr != self.learning_rate:
                self.learning_rate = new_lr
                print(f"Reduced learning rate to {self.learning_rate}")
            return

        # pct_improvement = (self.curr_loss - new_loss) / self.curr_loss
        # if pct_improvement <= self.lr_epsilon:
        #     self.learning_rate = self.learning_rate * 2
        #     print(f"Increased learning rate to {self.learning_rate}")
        #     return

    def train(self, X, Y, iterations):
        for i in range(iterations + 1):
            Y_hat = self.predict(X)
            new_loss = self.compute_loss(Y_hat, Y)
            if self.dynamic_lr:
                self.adjust_learning_rate(new_loss)
            if i % 100 == 0:
                print(f"Loss at iteration {i}: {new_loss}")

            self.apply_backprop(Y_hat, Y)
            self.curr_loss = new_loss

    def apply_backprop(self, Y_hat, Y):
        dloss_dy_hat = self.loss_cls.dloss_dy_hat(Y_hat, Y=Y)
        next_layer_dloss_dinput = dloss_dy_hat
        for layer in reversed(self.layers):
            X_out = self.layer_outputs[layer.id]["output"]
            Z_out = self.layer_outputs[layer.id]["Z"]
            X_in = self.layer_outputs[layer.id - 1]["output"]
            self.update_params(
                layer,
                next_layer_dloss_dinput,
                X_out,
                Z_out,
                X_in,
                Y_hat,
            )
            next_layer_dloss_dinput = layer.dloss_dinput(next_layer_dloss_dinput, X_out)

    def update_params(self, layer, next_layer_dloss_dinput, X_out, Z_out, X_in, Y_hat):
        dloss_dW = layer.dloss_dW(next_layer_dloss_dinput, X_out, Z_out, X_in)
        # dloss_dW = self.normalize_based_on_step_size(dloss_dW, next_layer_dloss_dinput)
        if self.loss_cls.regularization:
            dreg_loss_dW = self.loss_cls.dreg_loss_dparam(Y_hat, layer.W)
            dloss_dW += dreg_loss_dW
        layer.W -= self.learning_rate * dloss_dW

        dloss_db = layer.dloss_db(next_layer_dloss_dinput, X_out, Z_out, X_in)
        if self.loss_cls.regularization:
            dreg_loss_db = self.loss_cls.dreg_loss_dparam(Y_hat, layer.b)
            dloss_db += dreg_loss_db
        # dloss_db = self.normalize_based_on_step_size(dloss_db, next_layer_dloss_dinput)
        layer.b -= self.learning_rate * dloss_db

    def normalize_based_on_step_size(self, dparam, step_loss_size):
        """
        The idea here is that we know how far away we to hit a loss of 0
        We know how much loss we can reduce by moving 1 step in y_hat
        So to make sure we don't overshoot, we can approximate the amount
        of steps to ensure we don't overshoot the loss

        This was important for regression, to ensure we don't get large weights
        """
        max_step_size = max(1, self.curr_loss // np.sum(step_loss_size))
        dparam_sum = np.sum(dparam)
        dparam = dparam / dparam_sum
        return dparam * max_step_size

    def score(self, X, Y):
        prediction = (self.predict(X) >= 0.5).astype(int)
        num_correct = (Y == prediction).sum()
        print(f"Accuracy: {num_correct / X.shape[1]}")

    def compute_loss(self, Y_hat, Y) -> float:
        # Y_hat are the predictions
        return self.loss_cls.loss(Y_hat, Y, self.get_all_params())
