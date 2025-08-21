
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader


def prepare_data(x, y):



    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=42)


    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)


    x_train_t = torch.tensor(x_train_scaled, dtype=torch.float32)
    x_val_t = torch.tensor(x_val_scaled, dtype=torch.float32)
    x_test_t = torch.tensor(x_test_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    input_dim = x_train.shape[1]
    output_dim = len(set(y_train))


    train_ds = TensorDataset(x_train_t, y_train_t)
    val_ds = TensorDataset(x_val_t, y_val_t)
    test_ds = TensorDataset(x_test_t, y_test_t)

    return train_ds, val_ds, test_ds, input_dim, output_dim


def main():

    x, y = make_classification(n_samples=3000, n_features=50, n_classes=2, random_state=42)
    train_ds, val_ds, test_ds, inp_dim, op_dim = prepare_data(x, y)

    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)



    class DeepClassifier(nn.Module):
        def __init__(self, input_dim, hidden1, hidden2, output_dim, dropout=0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden2, output_dim)
            )

        def forward(self, x):
            return self.net(x)



    h1, h2, dropout, lr, epochs, patience, patience_counter = 64, 64, 0.2, 0.0001, 100, 10, 0
    best_val_acc = 0
    best_state = None

    model = DeepClassifier(inp_dim, h1, h2, op_dim, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()


        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch + 1}: Accuracy on validation set → {val_acc:.4f}")


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter > patience:
            print(f"Training stopped early at epoch {epoch + 1}")
            model.load_state_dict(best_state)
            break

    print(f"Highest Validation Accuracy Reached: {best_val_acc:.4f}")


    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    test_acc = correct / total
    print(f"Performance on Unseen Test Data: {test_acc:.4f}")

if __name__=="__main__":
    main()

