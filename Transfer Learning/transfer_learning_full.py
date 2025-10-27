import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prepare_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes

def get_model(num_classes=10):
    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for param in model.parameters():
        param.requires_grad = True  # fine-tune all layers
    return model

def train_model(model, trainloader, testloader, epochs=5, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        running_loss, running_corrects = 0.0, 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_acc = running_corrects.double() / len(trainloader.dataset)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc.item())
        print(f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        model.eval()
        running_loss, running_corrects = 0.0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(testloader.dataset)
        epoch_acc = running_corrects.double() / len(testloader.dataset)
        val_loss.append(epoch_loss)
        val_acc.append(epoch_acc.item())
        print(f"Val Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    return train_loss, train_acc, val_loss, val_acc

def main():
    print("Using device:", device)
    trainloader, testloader, classes = prepare_data()

    print("\n=== Transfer Learning WITH Grad (Fine-tuning of all layers) ===")
    model = get_model(num_classes=10)
    train_loss, train_acc, val_loss, val_acc = train_model(model, trainloader, testloader, epochs=5)

    plt.figure(figsize=(10, 5))
    plt.plot(train_acc, label="Train Acc (Fine-tuning)")
    plt.plot(val_acc, label="Val Acc (Fine-tuning)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Transfer Learning: Full Fine-tuning")
    plt.show()

if __name__ == "__main__":
    main()
