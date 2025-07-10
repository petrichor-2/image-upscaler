import os
import cv2
import random
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F

class AdvancedAugmentation:
    def __init__(self, augment_prob=0.7):
        self.augment_prob = augment_prob

    def __call__(self, lr_img, hr_img):
        """Apply identical augmentations to LR/HR pairs"""
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
            crop_size = int(min(h, w) * 0.8)  # Crop 80% of the image
            i = random.randint(0, h - crop_size)
            j = random.randint(0, w - crop_size)
            
            lr_img = F.resized_crop(lr_img, i, j, crop_size, crop_size, (h, w))
            hr_crop_size = crop_size * 4  # HR is 4x larger
            hr_i, hr_j = i * 4, j * 4
            hr_img = F.resized_crop(hr_img, hr_i, hr_j, hr_crop_size, hr_crop_size, (h*4, w*4))

        # Cutout augmentation
        if random.random() < 0.3:  # 30% chance
            lr_img = self._apply_cutout(lr_img)
            hr_img = self._apply_cutout(hr_img)

        return lr_img, hr_img

    def _apply_cutout(self, img):
        """Apply cutout augmentation to an image"""
        _, h, w = img.shape
        cutout_size = int(min(h, w) * 0.2)  # 20% of image size
        x = random.randint(0, w - cutout_size)
        y = random.randint(0, h - cutout_size)
        img = img.clone()
        img[:, y:y+cutout_size, x:x+cutout_size] = 0  # Set to black
        return img

def generate_downsampled_pairs(data_dir, force_rewrite=False):
    """Create HR (256x256) and LR (64x64) image pairs from source directory."""
    # Create output directories
    hr_folder = os.path.join(data_dir, "HR_256")
    lr_folder = os.path.join(data_dir, "LR_64")
    os.makedirs(hr_folder, exist_ok=True)
    os.makedirs(lr_folder, exist_ok=True)

    # Process each image file
    for filename in os.listdir(data_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        
        # Skip if files exist (unless force_rewrite=True)
        if not force_rewrite and os.path.exists(os.path.join(hr_folder, filename)) \
                           and os.path.exists(os.path.join(lr_folder, filename)):
            continue
            
        # Read and resize images
        image = cv2.imread(os.path.join(data_dir, filename))
        if image is None:
            continue
            
        high_res = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        low_res = cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC)
        
        # Save pairs
        cv2.imwrite(os.path.join(hr_folder, filename), high_res)
        cv2.imwrite(os.path.join(lr_folder, filename), low_res)

class PairDataset(Dataset):
    def __init__(self, data_dir, mode='train', split=[0.7, 0.2, 0.1], augment=False):
        """
        Args:
            data_dir: Path to parent directory containing HR_256 and LR_64 folders
            mode: 'train', 'val', or 'test'
            split: [train_ratio, val_ratio, test_ratio]
            augment: Whether to apply data augmentation
        """
        self.hr_dir = os.path.join(data_dir, "HR_256")
        self.lr_dir = os.path.join(data_dir, "LR_64")
        self.mode = mode
        all_files = sorted([f for f in os.listdir(self.hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        # Calculate split indices
        train_idx = int(len(all_files) * split[0])
        val_idx = train_idx + int(len(all_files) * split[1])
        
        # Select files based on mode
        if mode == 'train':
            self.filenames = all_files[:train_idx]
        elif mode == 'val':
            self.filenames = all_files[train_idx:val_idx]
        elif mode == 'test':
            self.filenames = all_files[val_idx:]
        else:
            raise ValueError("Mode must be 'train', 'val', or 'test'")

        self.augment = augment
        self.augmentor = AdvancedAugmentation() if augment else None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """Returns (LR_image, HR_image) pair as normalized tensors."""
        hr_img = cv2.imread(os.path.join(self.hr_dir, self.filenames[idx]))
        lr_img = cv2.imread(os.path.join(self.lr_dir, self.filenames[idx]))
        
        if hr_img is None or lr_img is None:
            return self.__getitem__(random.randint(0, len(self)-1))
            
        # Convert to tensor and normalize
        lr_img, hr_img = self.transform(lr_img), self.transform(hr_img)
        
        # Apply augmentations only during training
        if self.augment and self.augmentor and self.mode == 'train':
            lr_img, hr_img = self.augmentor(lr_img, hr_img)

        return lr_img, hr_img

def get_dataloaders(data_dir, batch_size=32, augment_train=True):
    """Returns train/val/test dataloaders."""
    # First generate the image pairs
    generate_downsampled_pairs(data_dir)
    
    # Create datasets
    datasets = {
        'train': PairDataset(data_dir, mode='train', augment=augment_train),
        'val': PairDataset(data_dir, mode='val'),
        'test': PairDataset(data_dir, mode='test')
    }
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(datasets['val'], batch_size=batch_size, num_workers=2),
        'test': DataLoader(datasets['test'], batch_size=batch_size, num_workers=2)
    }
    return dataloaders

if __name__ == "__main__":
    # Your specific path
    IMAGE_PATH = r"ADD_IMG_PATH_HERE"
    
    # Initialize dataloaders
    dataloaders = get_dataloaders(
        data_dir=IMAGE_PATH,
        batch_size=16,
        augment_train=True
    )
    
    # Test the dataloader
    print(f"Found {len(dataloaders['train'].dataset)} training images")
    print(f"Found {len(dataloaders['val'].dataset)} validation images")
    print(f"Found {len(dataloaders['test'].dataset)} test images")
    
    for lr, hr in dataloaders['train']:
        print("\nFirst batch:")
        print("LR shape:", lr.shape)  # Should be [batch, 3, 64, 64]
        print("HR shape:", hr.shape)  # Should be [batch, 3, 256, 256]
        
        # Visualize first image
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.title("Low Resolution")
        plt.imshow(lr[0].permute(1, 2, 0).numpy() * 0.5 + 0.5)  # Unnormalize
        
        plt.subplot(1, 2, 2)
        plt.title("High Resolution")
        plt.imshow(hr[0].permute(1, 2, 0).numpy() * 0.5 + 0.5)  # Unnormalize
        
        plt.show()
        break