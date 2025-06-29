import os
import cv2
import torch
import numpy as np
from torchvision import transforms

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
    files = os.listdir(os.path.join(data_dir, "HR_256"))
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
    def __init__(self, data_dir, type, split=[0.7, 0.2, 0.1]):
        # Save directory paths
        self.data_dir = data_dir
        self.hr_dir = os.path.join(data_dir, "HR_256")
        self.lr_dir = os.path.join(data_dir, "LR_64")

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
        self.filenames = os.listdir(self.hr_dir)[self.start_idx:self.end_idx]
        
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
        hr_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(hr_image))
        lr_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(lr_image))

        # low-res as image, high-res as label
        return lr_image, hr_image

def get_data_loaders(data_dir, batch_size=64, get_train=True, get_val=True, get_test=True):
    data_loaders = {}
    
    if get_train:
        train_dataset = PairDataset(data_dir, type='train')
        data_loaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if get_val:
        val_dataset = PairDataset(data_dir, type='val')
        data_loaders['val'] = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    if get_test:
        test_dataset = PairDataset(data_dir, type='test')
        data_loaders['test'] = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return data_loaders