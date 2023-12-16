import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Function to check if a point is in a specific range (you may need to define this)
def in_range(x, y) -> bool:
    return y < 0.3 * x + 2


def generate_train_data_classification(samples=10000):
    data = []
    labels = []
    for _ in range(samples):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        data.append((x, y))
        if in_range(x, y):
            labels.append(1)
        else:
            labels.append(0)
    return np.array(data), np.array(labels)


# Generate data
data, labels = generate_train_data_classification(samples=10000)

# Convert to PyTorch tensors
data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.float32)


# Define a simple neural network
class SimpleClassifier(nn.Module):
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
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


# Instantiate the model, loss function, and optimizer
model = SimpleClassifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Create a DataLoader for training
train_dataset = TensorDataset(data_tensor, labels_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_data, batch_labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels.view(-1, 1))
        loss.backward()
        optimizer.step()

# Evaluate accuracy on the entire dataset
model.eval()
with torch.no_grad():
    outputs = model(data_tensor)
    predicted_labels = (outputs >= 0.5).float()
    accuracy = (predicted_labels == labels_tensor.view(-1, 1)).sum().item() / len(
        labels
    )
    print(f"Accuracy: {accuracy * 100:.2f}%")
