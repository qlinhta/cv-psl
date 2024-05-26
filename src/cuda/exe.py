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
from torch.nn.parallel import DataParallel
from torch.backends import cudnn
from torch import nn, optim
from prettytable import PrettyTable
from tqdm import tqdm
from torchvision.utils import save_image
from tools import figure_train_val
from models import get_model_by_id

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


def save_model(model, epoch, model_name, best=False):
    model_dir = 'saved_models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    suffix = 'best' if best else f'epoch_{epoch}'
    model_path = os.path.join(model_dir, f"{model_name}_{suffix}.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")


def save_batch_images(images, labels, phase, num_images=8):
    dump_dir = os.path.join('dumps', phase)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    num_images_to_save = min(num_images, images.size(0))
    for i in range(num_images_to_save):
        save_image(images[i], os.path.join(dump_dir, f'image_{i}_label_{labels[i].item()}.png'))


def train_model(train_loader, val_loader, device, model_id, num_epochs=10):
    model_info = get_model_by_id(model_id)
    model = model_info.get_model().to(device)
    model_name = model_info.name

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5.728983638103915e-05)

    acc_train, acc_val = [], []
    loss_train, loss_val = [], []
    best_val_accuracy = 0.0

    train_images_saved = False
    val_images_saved = False

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training")
        for batch_idx, (images, labels) in enumerate(train_bar):
            if not train_images_saved:
                save_batch_images(images, labels, 'train')
                train_images_saved = True
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_bar.set_postfix(loss=running_loss / len(train_loader), accuracy=100 * correct / total)

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        loss_train.append(train_loss)
        acc_train.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation")
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_bar):
                if not val_images_saved:
                    save_batch_images(images, labels, 'val')
                    val_images_saved = True
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_bar.set_postfix(loss=val_loss / len(val_loader), accuracy=100 * correct / total)

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        loss_val.append(val_loss)
        acc_val.append(val_accuracy)

        table = PrettyTable()
        table.field_names = ["Epoch", "Train Loss", "Val Loss", "Train Accur.", "Val Accur."]
        table.add_row([epoch + 1, train_loss, val_loss, train_accuracy, val_accuracy])
        print(table)

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(model, epoch + 1, model_name, best=True)

    figure_train_val(model_name, acc_train, acc_val, loss_train, loss_val, save=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train bird classification model")
    parser.add_argument('--train_csv', type=str, required=True, help="Path to the train labels CSV file")
    parser.add_argument('--val_csv', type=str, required=True, help="Path to the validation labels CSV file")
    parser.add_argument('--train_dir', type=str, required=True, help="Path to the train images directory")
    parser.add_argument('--val_dir', type=str, required=True, help="Path to the validation images directory")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for the dataloaders")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for the dataloaders")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--model_id', type=int, required=True, help="ID of the model to use")
    parser.add_argument('--num_classes', type=int, default=30, help="Number of output classes")

    args = parser.parse_args()

    logger.info("Using model ID: {}".format(args.model_id))
    model_info = get_model_by_id(args.model_id)
    logger.info("Using model: {}".format(model_info.name))
    logger.info("Number of output classes: {}".format(args.num_classes))
    logger.info("Number of epochs: {}".format(args.num_epochs))
    logger.info("Batch size: {}".format(args.batch_size))
    logger.info("Number of workers: {}".format(args.num_workers))

    train_transform, val_transform = augment()

    train_loader, val_loader = loader(
        args.train_csv,
        args.val_csv,
        args.train_dir,
        args.val_dir,
        args.batch_size,
        args.num_workers,
        train_transform,
        val_transform
    )

    device = get_device()

    train_model(train_loader, val_loader, device, args.model_id, args.num_epochs)
