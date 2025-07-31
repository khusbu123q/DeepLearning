import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings

warnings.filterwarnings("ignore")
plt.ion()

# ===== Helper to show image + landmarks
def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, c='r', marker='.')
    plt.axis('off')
    plt.tight_layout()
    plt.pause(0.001)

# ===== Custom Transforms
class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w), anti_aliasing=True)
        landmarks = landmarks * [new_w / w, new_h / h]
        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
    def __init__(self, output_size):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        image = image[top: top + new_h, left: left + new_w]
        landmarks = landmarks - [left, top]
        return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = image.transpose((2, 0, 1))  # HWC → CHW
        return {'image': torch.from_numpy(image).float(),
                'landmarks': torch.from_numpy(landmarks).float()}

# ===== Dataset
class FaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].values.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}
        if self.transform:
            sample = self.transform(sample)
        return sample

# ===== Transform pipeline
composed = transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()])

csv_path = 'faces/face_landmarks.csv'
img_dir = 'faces/'
assert os.path.exists(csv_path), "CSV file not found"
face_dataset = FaceLandmarksDataset(csv_file=csv_path, root_dir=img_dir, transform=composed)

# ===== Visualize first 4
fig = plt.figure(figsize=(10, 5))
for i in range(4):
    sample = face_dataset[i]
    img = sample['image'].numpy().transpose((1, 2, 0))  # CHW → HWC
    show_landmarks(img, sample['landmarks'].numpy())
    plt.subplot(1, 4, i + 1).set_title(f"Sample #{i}")
plt.ioff()
plt.show()

# ===== Test transforms individually
scale = Rescale(256)
crop = RandomCrop(128)
composed_only = transforms.Compose([Rescale(256), RandomCrop(224)])
sample = face_dataset[65]
sample = {'image': sample['image'].numpy().transpose((1, 2, 0)), 'landmarks': sample['landmarks'].numpy()}

fig = plt.figure(figsize=(12, 4))
for i, tsfrm in enumerate([scale, crop, composed_only]):
    transformed = tsfrm(sample)
    ax = plt.subplot(1, 3, i + 1)
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(transformed['image'], transformed['landmarks'])
plt.show()

# ===== DataLoader
transformed_dataset = FaceLandmarksDataset(csv_file=csv_path, root_dir=img_dir, transform=composed)
dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=0)

def show_landmarks_batch(sample_batched):
    images_batch = sample_batched['image']
    landmarks_batch = sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')
    plt.title("Batch from dataloader")

# Show 1 batch
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(), sample_batched['landmarks'].size())
    if i_batch == 0:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.show()
        break

# ===== OPTIONAL: Image Classification Dataset (Comment if not needed)
# data_transform = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])
# hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train', transform=data_transform)
# dataset_loader = DataLoader(hymenoptera_dataset, batch_size=4, shuffle=True, num_workers=0)
