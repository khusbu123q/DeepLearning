import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Data Loading ------------------
def data_load():
    data = np.load('1000G_reqnorm_float64.npy')

    num_genes, num_samples = data.shape
    print(f"Number of genes (rows): {num_genes}, samples (cols): {num_samples}")

    # Z-score normalization
    data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-3)

    num_lm = 943
    X = data[:num_lm, :].T  # samples × landmark genes
    Y = data[num_lm:, :].T  # samples × target genes

    # Shuffle
    idx = np.random.permutation(X.shape[0])
    X, Y = X[idx], Y[idx]

    # Split train/val/test
    n = X.shape[0]
    train_end, val_end = int(0.7*n), int(0.9*n)

    def to_ds(x, y):
        return TensorDataset(torch.tensor(x, dtype=torch.float32),
                             torch.tensor(y, dtype=torch.float32))

    return (to_ds(X[:train_end], Y[:train_end]),
            to_ds(X[train_end:val_end], Y[train_end:val_end]),
            to_ds(X[val_end:], Y[val_end:])), X.shape[1], Y.shape[1]


# ------------------ Model ------------------
class GenePredictionFFN(nn.Module):
    def __init__(self, input_dim, output_dim, h1=700, h2=600, h3=500, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, h3), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h3, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


# ------------------ Training ------------------
def train_epoch(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
    return total_loss / len(dataloader.dataset)


def eval_epoch(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item() * X.size(0)
    return total_loss / len(dataloader.dataset)


# ------------------ Main ------------------
def main():
    (train_ds, val_ds, test_ds), input_dim, output_dim = data_load()
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=32, drop_last=True)

    model = GenePredictionFFN(input_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    n_epochs = 200
    train_losses, val_losses = [], []

    for epoch in range(n_epochs):
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        val_loss = eval_epoch(val_loader, model, loss_fn)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}: Train {train_loss:.4f}, Val {val_loss:.4f}")

    # Plot losses
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

    # Save & test
    torch.save(model.state_dict(), "gene_pred_model.pth")
    print("✅ Model saved to gene_pred_model.pth")

    final_test_loss = eval_epoch(test_loader, model, loss_fn)
    print(f"Final Test Loss: {final_test_loss:.4f}")


if __name__ == "__main__":
    main()
