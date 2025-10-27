import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_feature_extractor():
    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    model.fc = nn.Identity()
    model.eval()
    return model

def extract_features(model, dataloader):
    features, labels = [], []
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels.append(lbls.numpy())
    return np.vstack(features), np.hstack(labels)

def prepare_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes

def train_svm(X_train, y_train):
    svm_clf = SVC(kernel="rbf", C=10, random_state=42)
    svm_clf.fit(X_train, y_train)
    return svm_clf

def evaluate_model(svm_clf, X_test, y_test, classes):
    y_pred = svm_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\nSVM Accuracy on CIFAR-10 test set:", acc)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def main():
    print("Using device:", device)
    trainloader, testloader, classes = prepare_data()

    print("\n=== Extracting Features with ResNet18 ===")
    model = get_feature_extractor()
    X_train, y_train = extract_features(model, trainloader)
    X_test, y_test = extract_features(model, testloader)

    print("Train Features Shape:", X_train.shape)
    print("Test Features Shape:", X_test.shape)
    print("\nNumber of training samples:", X_train.shape[0])
    print("Number of features per sample:", X_train.shape[1])

    print("\n=== Training SVM on ResNet Features ===")
    svm_clf = train_svm(X_train, y_train)
    evaluate_model(svm_clf, X_test, y_test, classes)

if __name__ == "__main__":
    main()
