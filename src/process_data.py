import os
import cv2
import torch
import numpy as np
import random
from torchvision import transforms
import torchvision.transforms.functional as F


def generate_downsampled_pairs(data_dir, force_rewrite=False):
    # Make directories
    hr_folder = os.path.join(data_dir, "HR_256")
    lr_folder = os.path.join(data_dir, "LR_64")
    os.makedirs(hr_folder, exist_ok=True)
    os.makedirs(lr_folder, exist_ok=True)

    # Resize and save image pairs
    for filename in os.listdir(data_dir):
        if not filename.lower().endswith('.png'):
            continue
        image = cv2.imread(os.path.join(data_dir, filename))
        high_res_image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_CUBIC)
        low_res_image = cv2.resize(image, (64, 64), interpolation = cv2.INTER_CUBIC)
        if not force_rewrite and os.path.exists(os.path.join(hr_folder, filename)) \
                             and os.path.exists(os.path.join(lr_folder, filename)):
            continue
        cv2.imwrite(os.path.join(hr_folder, filename), high_res_image)
        cv2.imwrite(os.path.join(lr_folder, filename), low_res_image)

def get_split_indices(data_dir, split):
    files = sorted(os.listdir(os.path.join(data_dir, "HR_256")))
    train_index = int(len(files) * split[0])
    val_index = int(len(files) * (split[0] + split[1]))

    # Ensure that patients are not split between sets
    # By rounding to the neares 000
    while files[train_index].split('_')[1] != '000.png':
        train_index += 1
    while files[val_index].split('_')[1] != '000.png':
        val_index += 1

    return train_index, val_index

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, type, split=[0.7, 0.2, 0.1], augment=False):
        # Save directory paths
        self.data_dir = data_dir
        self.hr_dir = os.path.join(data_dir, "HR_256")
        self.lr_dir = os.path.join(data_dir, "LR_64")

        # Augmentation
        self.augment = augment
        self.augmentor = AdvancedAugmentation() if augment else None

        # Basic conversion transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.type = type

        # Get indices to split at
        train_idx, val_idx = get_split_indices(data_dir, split)
        if (type == 'train'):
            self.start_idx = 0
            self.end_idx = train_idx
        elif (type == 'val'):
            self.start_idx = train_idx
            self.end_idx = val_idx
        elif (type == 'test'):
            self.start_idx = val_idx
            self.end_idx = len(os.listdir(os.path.join(data_dir, "HR_256")))
        else:
            raise ValueError("Type must be 'train', 'val', or 'test'.")
        
        # Save filenames in list for easy access later
        # May want to sort for deterministic behavior
        self.filenames = sorted(os.listdir(self.hr_dir))[self.start_idx:self.end_idx]
        
    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        # Read the image pair
        hr_image = cv2.imread(os.path.join(self.hr_dir, self.filenames[idx]))
        lr_image = cv2.imread(os.path.join(self.lr_dir, self.filenames[idx]))

        if hr_image is None or lr_image is None:
            raise FileNotFoundError(f"Image not found at ",
                                    os.path.join(self.hr_dir, self.filenames[idx]))
        
        # Convert images to numpy arrays
        hr_image = np.array(hr_image)
        lr_image = np.array(lr_image)

        # Convert images to tensor and normalize to [-1, 1]
        hr_image = self.transform(hr_image)
        lr_image = self.transform(lr_image)

        # Apply augmentation only for training
        if self.augment and self.type == 'train':
            lr_image, hr_image = self.augmentor(lr_image, hr_image)

        # low-res as image, high-res as label
        return lr_image, hr_image

class AdvancedAugmentation:
    def __init__(self, augment_prob=0.7):
        self.augment_prob = augment_prob

    def __call__(self, lr_img, hr_img):
        if random.random() > self.augment_prob:
            return lr_img, hr_img

        # Random horizontal flip
        if random.random() > 0.5:
            lr_img = F.hflip(lr_img)
            hr_img = F.hflip(hr_img)

        # Random rotation (90° increments)
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            lr_img = F.rotate(lr_img, angle)
            hr_img = F.rotate(hr_img, angle)

        # Random crop and resize
        if random.random() > 0.5:
            _, h, w = lr_img.shape
            crop_size = int(min(h, w) * 0.8)
            i = random.randint(0, h - crop_size)
            j = random.randint(0, w - crop_size)
            lr_img = F.resized_crop(lr_img, i, j, crop_size, crop_size, (h, w))
            hr_crop_size = crop_size * 4
            hr_i, hr_j = i * 4, j * 4
            hr_img = F.resized_crop(hr_img, hr_i, hr_j, hr_crop_size, hr_crop_size, (h*4, w*4))

        # Cutout augmentation
        if random.random() < 0.3:
            lr_img = self._apply_cutout(lr_img)
            hr_img = self._apply_cutout(hr_img)

        return lr_img, hr_img

    def _apply_cutout(self, img):
        _, h, w = img.shape
        cutout_size = int(min(h, w) * 0.2)
        x = random.randint(0, w - cutout_size)
        y = random.randint(0, h - cutout_size)
        img = img.clone()
        img[:, y:y+cutout_size, x:x+cutout_size] = 0
        return img

def get_data_loaders(data_dir, batch_size=64, get_train=True, get_val=True, get_test=True, augment_train=True):
    data_loaders = {}
    
    if get_train:
        train_dataset = PairDataset(data_dir, type='train', augment=augment_train)
        data_loaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if get_val:
        val_dataset = PairDataset(data_dir, type='val')
        data_loaders['val'] = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    if get_test:
        test_dataset = PairDataset(data_dir, type='test')
        data_loaders['test'] = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return data_loaders