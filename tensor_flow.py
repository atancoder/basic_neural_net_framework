from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy  # type: ignore
from tensorflow.keras.optimizers.legacy import Adam  # type: ignore


def predict_with_threshold(probabilities, threshold=0.5):
    return (probabilities > threshold).astype(int)


def neural_net(training_data, training_labels, iterations=100):
    training_data = training_data.reshape(-1, 2)
    training_labels = training_labels.reshape(-1, 1)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                units=16,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-10),
            ),
            tf.keras.layers.Dense(
                units=8,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-10),
            ),
            tf.keras.layers.Dense(
                units=4,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-10),
            ),
            tf.keras.layers.Dense(
                units=2,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-10),
            ),
            tf.keras.layers.Dense(units=1, activation="sigmoid"),
        ]
    )
    model.compile(
        loss=BinaryCrossentropy(from_logits=False),
        optimizer=Adam(learning_rate=0.01),
    )
    model.fit(training_data, training_labels, epochs=iterations)
    probabilities = model.predict(training_data)
    num_correct = (training_labels == predict_with_threshold(probabilities)).sum()
    print(f"Accuracy: {num_correct / training_data.shape[0]}")


def print_weights(model):
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        print(f"Layer {i} weights: \n{weights[0]}")
        print(f"Layer {i} biases: \n{weights[1]}")
