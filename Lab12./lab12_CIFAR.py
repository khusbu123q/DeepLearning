import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import random


def fix_randomness(seed_val=64):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
NUM_EPOCHS = 2
NUM_FINAL_EPOCHS = 5
LR = 0.001
MOMENTUM = 0.9
CONV_KERNEL_SIZES = [5] + [3]*7
POOL_KERNEL = 2
NUM_LAYERS_START = 1
NUM_LAYERS_STOP = 8
BEST_LAYER_COUNT = 1
DATA_PATH = "../data"
RANDOM_SEED = 42

fix_randomness(RANDOM_SEED)

dataset_raw = torchvision.datasets.CIFAR10(
    root=DATA_PATH, train=True, transform=transforms.ToTensor(), download=True
)
loader_stats = DataLoader(dataset_raw, batch_size=5000, shuffle=False)

mean = 0
std = 0
num_batches = 0
for data, target in loader_stats:
    bs = data.size(0)
    data = data.view(bs, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    num_batches += bs
mean /= num_batches
std /= num_batches
mean = mean.numpy()
std = std.numpy()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_set = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, transform=transform, download=True)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_set = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, transform=transform, download=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

class_labels = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')


def visualize(img):
    img = img.cpu()
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = img * std_t + mean_t
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()


class ConvNet(nn.Module):
    def __init__(self, depth=2):
        super().__init__()
        in_ch = 3
        layers = []
        for i in range(depth):
            out_ch = 6 * (2 ** i) if i < depth - 1 else 16
            k = CONV_KERNEL_SIZES[i]
            pad = k // 2
            layers.append(nn.Conv2d(in_ch, out_ch, k, padding=pad))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(POOL_KERNEL, POOL_KERNEL-1))
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            feat = self.conv_layers(dummy)
            self.flat_size = feat.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flat_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.flat_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def run_training(model, loader, opt, criterion, epoch):
    model.train()
    loss_accum = 0.0
    num_batches = 0
    for idx, (data, labels) in enumerate(loader):
        data, labels = data.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        preds = model(data)
        loss = criterion(preds, labels)
        loss.backward()
        opt.step()
        loss_accum += loss.item()
        num_batches += 1
        if idx % 200 == 199:
            print(f"--> Epoch {epoch+1}, step {idx+1}: avg loss {loss_accum/200:.4f}")
            loss_accum = 0.0
    return loss_accum / max(1, num_batches)


def compute_loss(model, loader, criterion):
    model.eval()
    total_loss = 0
    batches = 0
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            preds = model(data)
            total_loss += criterion(preds, labels).item()
            batches += 1
    return total_loss / batches


def assess_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    per_class_correct = {cls: 0 for cls in class_labels}
    per_class_total = {cls: 0 for cls in class_labels}

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            preds = model(data)
            _, chosen = torch.max(preds, 1)
            total += labels.size(0)
            correct += (chosen == labels).sum().item()
            for lbl, pred in zip(labels, chosen):
                if lbl == pred:
                    per_class_correct[class_labels[lbl]] += 1
                per_class_total[class_labels[lbl]] += 1

    acc = 100 * correct / total
    print(f"\n>>> Final Accuracy on Test Set: {acc:.2f}% <<<")
    print("Category-wise breakdown:")
    for cls in class_labels:
        acc_cls = 100 * per_class_correct[cls] / per_class_total[cls]
        print(f" - {cls:10s}: {acc_cls:.2f}%")
    return acc, per_class_correct, per_class_total


def main():
    chosen_depths = list(range(NUM_LAYERS_START, NUM_LAYERS_STOP+1))
    train_errs = []
    val_errs = []

    model_final = ConvNet(depth=BEST_LAYER_COUNT).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_final.parameters(), lr=LR, momentum=MOMENTUM)

    for ep in range(NUM_FINAL_EPOCHS):
        run_training(model_final, train_loader, optimizer, loss_fn, ep)


    data_iter = iter(test_loader)
    imgs, lbls = next(data_iter)
    visualize(torchvision.utils.make_grid(imgs))
    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
    model_final.eval()
    outs = model_final(imgs)
    _, preds = torch.max(outs, 1)
    print("True labels:   ", ' | '.join(f"{class_labels[lbls[j]]}" for j in range(len(lbls))))
    print("Model guesses: ", ' | '.join(f"{class_labels[preds[j]]}" for j in range(len(preds))))


    assess_accuracy(model_final, test_loader)


if __name__ == "__main__":
    main()
