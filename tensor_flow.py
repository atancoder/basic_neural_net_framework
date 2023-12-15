from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy  # type: ignore
from tensorflow.keras.optimizers.legacy import Adam  # type: ignore


def predict_with_threshold(model, input_data, threshold=0.5):
    probabilities = tf.nn.sigmoid(model.predict(input_data)).numpy()
    return (probabilities > threshold).astype(int)


def relu_neural_net(training_data, training_labels):
    training_data = training_data.reshape(-1, 2)
    training_labels = training_labels.reshape(-1, 1)

    model = tf.keras.Sequential(
        [
            # tf.keras.layers.Dense(
            #     units=120,
            #     activation="relu",
            #     kernel_regularizer=tf.keras.regularizers.l2(1e-10),
            # ),
            # tf.keras.layers.Dense(
            #     units=40,
            #     activation="sigmoid",
            #     kernel_regularizer=tf.keras.regularizers.l2(1e-10),
            # ),
            # tf.keras.layers.Dense(
            #     units=20,
            #     activation="relu",
            #     kernel_regularizer=tf.keras.regularizers.l2(1e-10),
            # ),
            tf.keras.layers.Dense(units=1, activation="sigmoid"),
        ]
    )
    model.compile(
        loss=BinaryCrossentropy(from_logits=True),
        optimizer=Adam(learning_rate=0.01),
    )

    model.fit(training_data, training_labels, epochs=100)
    predictions = predict_with_threshold(model, training_data)
    num_correct = (predictions == predictions).sum()
    print(f"Accuracy: {num_correct / training_data.shape[0]}")
    print_weights(model)


def print_weights(model):
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        print(f"Layer {i} weights: \n{weights[0]}")
        print(f"Layer {i} biases: \n{weights[1]}")
