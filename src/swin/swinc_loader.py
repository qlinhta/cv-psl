import torch
import torch.nn as nn
import torch.optim as optim
from transformers import CLIPModel, CLIPProcessor
import timm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s [%(levelname)s] %(message)s')


class BirdDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, text_file=None):
        logger.info(f"Loading dataset from {csv_file}")
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.text_df = pd.read_csv(text_file) if text_file else None
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = self.labels_df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image=np.array(image))['image']

        if self.text_df is not None:
            text = self.text_df.iloc[idx, 1]
            text_inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            text_inputs = {key: val.squeeze(0) for key, val in text_inputs.items()}  # Remove batch dimension
            return image, label, text_inputs
        else:
            return image, label


def augment():
    logger.info("Creating data augmentation transforms")
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return train_transform, val_transform


def collate_fn(batch):
    images, labels, text_inputs = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.tensor(labels)
    input_ids = torch.nn.utils.rnn.pad_sequence([item['input_ids'] for item in text_inputs],
                                                batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence([item['attention_mask'] for item in text_inputs],
                                                     batch_first=True, padding_value=0)
    text_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    return images, labels, text_inputs


def loader(train_csv, val_csv, train_dir, val_dir, train_text, val_text, batch_size, num_workers, train_transform,
           val_transform):
    logger.info("Loading data")
    train_dataset = BirdDataset(csv_file=train_csv, root_dir=train_dir, transform=train_transform, text_file=train_text)
    val_dataset = BirdDataset(csv_file=val_csv, root_dir=val_dir, transform=val_transform, text_file=val_text)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, prefetch_factor=8, persistent_workers=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                            prefetch_factor=8, persistent_workers=True, collate_fn=collate_fn)

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
