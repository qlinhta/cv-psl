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

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s [%(levelname)s] %(message)s')


class BirdDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        logger.info(f"Loading dataset from {csv_file}")
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = self.labels_df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image=np.array(image))['image']

        return image, label


def augment():
    logger.info("Creating data augmentation transforms")
    train_transform = A.Compose([
        # A.Resize(256, 256),
        A.Resize(224, 224),
        A.RandomResizedCrop(224, 224),
        A.HorizontalFlip(),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_test_transform = A.Compose([
        # A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return train_transform, val_test_transform


def loader(train_csv, val_csv, train_dir, val_dir, batch_size, num_workers, train_transform, val_transform):
    logger.info("Loading data")
    train_dataset = BirdDataset(csv_file=train_csv, root_dir=train_dir, transform=train_transform)
    val_dataset = BirdDataset(csv_file=val_csv, root_dir=val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                            prefetch_factor=4, persistent_workers=True)

    return train_loader, val_loader


def device():
    logger.info("Checking for available devices")
    if torch.cuda.is_available():
        logger.info("Using CUDA")
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        logger.info("Using MPS")
        return torch.device('mps')
    else:
        logger.info("Using CPU")
        return torch.device('cpu')
