from pathlib import Path
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from collections import Counter

SVHN_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    transforms.Normalize(mean=[0.437, 0.404, 0.443], std=[0.243, 0.239, 0.242])
])

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.437, 0.404, 0.443], std=[0.243, 0.239, 0.242])
])

def load_labels(label_file_path):
    labels_dict = {}
    with open(label_file_path, 'r') as f:
        for line in f:
            img_name, label = line.strip().split()
            labels_dict[img_name] = int(label)
    return labels_dict

class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=SVHN_TRANSFORM, oversample_digits=None, max_images=1000):
        self.image_paths = sorted(list(Path(folder_path).glob("*.png")))
        label_file_path = os.path.join(folder_path, "labels.txt")
        self.labels_dict = load_labels(label_file_path)
        self.image_paths = [p for p in self.image_paths if p.name in self.labels_dict]
        self.labels = [self.labels_dict[p.name] for p in self.image_paths]

        if max_images and len(self.image_paths) > max_images:
            self.image_paths = self.image_paths[:max_images]
            self.labels = self.labels[:max_images]

        if oversample_digits:
            oversampled_paths = []
            oversampled_labels = []
            for img_path, label in zip(self.image_paths, self.labels):
                oversampled_paths.append(img_path)
                oversampled_labels.append(label)
                if label in oversample_digits:
                    oversampled_paths.extend([img_path] * oversample_digits[label])
                    oversampled_labels.extend([label] * oversample_digits[label])
            self.image_paths = oversampled_paths
            self.labels = oversampled_labels

        self.transform = transform
        print(f"Loaded {len(self.image_paths)} images with labels: {Counter(self.labels)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def load_data(data_dir="Dataset", batch_size=32):
    oversample_digits = {1: 2, 7: 1}
    train_dataset = CustomDataset(os.path.join(data_dir, "train"), transform=SVHN_TRANSFORM,
                                  oversample_digits=oversample_digits, max_images=1000)
    test_dataset = CustomDataset(os.path.join(data_dir, "test"), transform=INFERENCE_TRANSFORM,
                                 oversample_digits=None, max_images=1000)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), \
           DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return INFERENCE_TRANSFORM(image).unsqueeze(0), None
