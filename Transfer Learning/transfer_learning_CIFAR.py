

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Data Preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')



def get_model(finetune=True):

    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)


    model.fc = nn.Linear(512, 10).to(device)

    if not finetune:
        # Freeze all except last fc
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    return model



# Training Function
def train_model(model, trainloader, testloader, epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                  model.parameters()), lr=lr)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        # ---- Training ----
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

        # ---- Validation ----
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



print("\n=== Transfer Learning WITHOUT Grad (Feature Extraction) ===")
model_feat = get_model(finetune=False)
train_loss_feat, train_acc_feat, val_loss_feat, val_acc_feat = train_model(model_feat, trainloader, testloader, epochs=5)

# Run Transfer Learning (Fine-tuning)
print("\n=== Transfer Learning WITH Grad (Fine-tuning) ===")
model_fine = get_model(finetune=True)
train_loss_fine, train_acc_fine, val_loss_fine, val_acc_fine = train_model(model_fine, trainloader, testloader, epochs=5)


plt.figure(figsize=(10, 5))
plt.plot(train_acc_feat, label="Train Acc (Feature Extraction)")
plt.plot(val_acc_feat, label="Val Acc (Feature Extraction)")
plt.plot(train_acc_fine, label="Train Acc (Fine-tuning)")
plt.plot(val_acc_fine, label="Val Acc (Fine-tuning)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Transfer Learning: Feature Extraction vs Fine-tuning")
plt.show()



