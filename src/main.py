import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from builder import loader, augment, device
from tools import figure_train_val
from tqdm import tqdm
from prettytable import PrettyTable
import logging
import coloredlogs
import os
from models import get_model_by_id
from torchvision.utils import save_image

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s [%(levelname)s] %(message)s')


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

        if val_accuracy > best_val_accuracy:
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

    device = device()

    train_model(train_loader, val_loader, device, args.model_id, args.num_epochs)
