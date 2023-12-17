import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cpu"
BATCH_SIZE = 64
import numpy as np

# seed 7 leads to dead neurons
torch.manual_seed(6)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
        )

    def forward(self, x):
        self.intermediate_outputs = []  # Reset the list for each forward pass
        for layer in self.linear_relu_stack:
            x = layer(x)
            self.intermediate_outputs.append(x.clone())
        return x


def neural_net(X, Y, iterations):
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    dataset = TensorDataset(X, Y)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    model = NeuralNetwork().to(DEVICE)
    test(model, X, Y)
    train(model, data_loader, iterations)
    test(model, X, Y)


def test(model, X, Y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probabilities = torch.sigmoid(logits)
        predicted_labels = (probabilities >= 0.5).float()
        accuracy = (predicted_labels == Y.view(-1, 1)).sum().item() / len(Y)
        print(f"Accuracy: {accuracy * 100:.2f}%")


def train(model, data_loader, iterations):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()  # Sets the model into training mode
    for iteration in range(iterations):
        for X, y in data_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        loss = loss.item()
        print(f"{iteration}/{iterations}: loss: {loss:>7f}")
