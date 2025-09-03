import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.optim as optim
import torchvision.transforms as transforms


# Download training data from open datasets.
training_data = torchvision.datasets.MNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor(),
)

batch_size = 64  # no of batches

# Create data loaders.
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# For data shape and type
for X, y in test_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Define the transformation to convert PIL Image to tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),                  # Converts PIL Image to tensor
    transforms.Normalize((0.5,), (0.5,))   # Normalize tensor with mean=0.5 and std=0.5
])


# Define a CNN architecture
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)  # flatten
        x = self.dropout(self.relu(self.fc1(x)))  # fully connected layer
        x = self.fc2(x)
        return x


# Training loop
def train_model(model, train_loader, criterion, optimizer, epochs=10, device="cpu"):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss_val = criterion(outputs, labels)
            loss_val.backward()
            optimizer.step()
            running_loss += loss_val.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")


# Testing loop
def evaluate_model(model, test_loader, criterion, device="cpu"):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss = criterion(outputs, labels)
            total_loss += test_loss.item()
            preds = outputs.argmax(dim=1, keepdims=True)
            correct += preds.eq(labels.view_as(preds)).sum().item()
    accuracy = 100 * correct / len(test_loader.dataset)
    print(f"Test Loss: {total_loss / len(test_loader):.4f}, Accuracy: {accuracy:.2f}%")


# Main function
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n--- Training ---")
    train_model(model, train_loader, criterion, optimizer, epochs=10, device=device)

    print("\n--- Testing ---")
    evaluate_model(model, test_loader, criterion, device=device)


if __name__ == "__main__":
    main()
