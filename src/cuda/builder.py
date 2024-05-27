import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import logging
import coloredlogs
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cuda_transforms
from torch.backends import cudnn

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s [%(levelname)s] %(message)s')


class BirdDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, training=True):
        logger.info(f"Loading dataset from {csv_file}")
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = self.labels_df.iloc[idx, 1]

        image = np.array(image)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().cuda()  # Convert to CUDA tensor

        if self.training:
            image_tensor = cuda_transforms.random_crop(image_tensor, 224, 224)
            if np.random.rand() > 0.5:
                image_tensor = cuda_transforms.horizontal_flip(image_tensor)
            image_tensor = cuda_transforms.gaussian_noise(image_tensor, 0.0, 0.1)
        image_tensor = cuda_transforms.normalize(image_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        if self.transform:
            image = self.transform(image=image_tensor)['image']

        return image, label


def augment():
    logger.info("Creating data augmentation transforms")
    train_transform = A.Compose([
        ToTensorV2()
    ])

    val_transform = A.Compose([
        ToTensorV2()
    ])

    return train_transform, val_transform


def loader(train_csv, val_csv, train_dir, val_dir, batch_size, num_workers, train_transform, val_transform):
    logger.info("Loading data")
    train_dataset = BirdDataset(csv_file=train_csv, root_dir=train_dir, transform=train_transform, training=True)
    val_dataset = BirdDataset(csv_file=val_csv, root_dir=val_dir, transform=val_transform, training=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                            prefetch_factor=4, persistent_workers=True)

    return train_loader, val_loader


def get_device():
    logger.info("Checking for available devices")
    if torch.cuda.is_available():
        logger.info("Using CUDA")
        cudnn.benchmark = True
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        logger.info("Using MPS")
        return torch.device('mps')
    else:
        logger.info("Using CPU")
        return torch.device('cpu')
