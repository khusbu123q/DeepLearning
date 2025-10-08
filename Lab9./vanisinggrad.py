import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score

torch.manual_seed(42)
np.random.seed(42)

# Make data: Two circles on x-y plane as a classification problem
X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y.numpy().reshape(-1))
plt.show()


class MLP(nn.Module):
    def __init__(self, activation, n_hidden=5, n_layers=3):
        super().__init__()
        activations = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh()
        }
        layers = [nn.Linear(2, n_hidden)]
        layers.append(activations[activation])
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(activations[activation])
        layers.append(nn.Linear(n_hidden, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_model(model, X, y, n_epochs=100, batch_size=32):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    loss_history = []
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(X.size(0))
        for i in range(0, X.size(0), batch_size):
            idx = perm[i:i + batch_size]
            X_batch, y_batch = X[idx], y[idx]
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            y_pred = model(X)
            acc = accuracy_score(y.numpy().reshape(-1), (y_pred.numpy() > 0.5).astype(int))
            loss_history.append(loss.item())
    return loss_history


def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        acc = accuracy_score(y.numpy().reshape(-1), (y_pred.numpy() > 0.5).astype(int))
    return acc


class WeightCapture:
    def __init__(self, model):
        self.model = model
        self.weights = []
        self.epochs = []

    def capture(self, epoch):
        weight = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                weight[name] = param.detach().numpy().copy()
        self.weights.append(weight)
        self.epochs.append(epoch)


def plotweight(capture_cb):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 10), constrained_layout=True)
    ax[0].set_title("Mean weight")
    for key in capture_cb.weights[0]:  # Use the first weight to get keys
        ax[0].plot(capture_cb.epochs, [w[key].mean() for w in capture_cb.weights], label=key)
    ax[0].legend()
    ax[1].set_title("S.D.")
    for key in capture_cb.weights[0]:  # Use the first weight to get keys
        ax[1].plot(capture_cb.epochs, [w[key].std() for w in capture_cb.weights], label=key)
    ax[1].legend()
    plt.show()


def full_train(X, y, activation="relu", n_epochs=100, batch_size=32):
    model = MLP(activation=activation, n_layers=5)
    capture_cb = WeightCapture(model)
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters())
    loss_history = []
    gradhistory = []

    # Before training
    capture_cb.capture(-1)
    print("Before training: Accuracy", evaluate_model(model, X, y))

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(X.size(0))
        for i in range(0, X.size(0), batch_size):
            idx = perm[i:i + batch_size]
            X_batch, y_batch = X[idx], y[idx]
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            # Capture gradients only for first batch per epoch
            if i == 0:
                grad_data = {}
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        grad_data[name] = param.grad.detach().numpy().copy()
                gradhistory.append(grad_data)
                loss_history.append(loss.item())

        capture_cb.capture(epoch)

    # After training
    print("After training: Accuracy", evaluate_model(model, X, y))
    plotweight(capture_cb)

    return gradhistory, loss_history


def plot_gradient(gradhistory, losshistory):
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 12), constrained_layout=True)

    # Plot mean gradient
    ax[0].set_title("Mean gradient")
    for key in gradhistory[0]:  # Use the first gradient to get keys
        ax[0].plot(range(len(gradhistory)), [g[key].mean() for g in gradhistory], label=key)
    ax[0].legend()

    # Plot std gradient (log scale)
    ax[1].set_title("S.D. of gradient")
    for key in gradhistory[0]:  # Use the first gradient to get keys
        ax[1].semilogy(range(len(gradhistory)), [g[key].std() for g in gradhistory], label=key)  # Use semilogy for better scale
    ax[1].legend()

    # Plot loss
    ax[2].set_title("Loss")
    ax[2].plot(range(len(losshistory)), losshistory)
    ax[2].set_xlabel("Epoch")

    plt.show()


for activation in ["sigmoid", "tanh", "relu"]:
    print(f"\n--- Activation: {activation} ---")
    gradhistory, losshistory = full_train(X, y, activation=activation)
    plot_gradient(gradhistory, losshistory)