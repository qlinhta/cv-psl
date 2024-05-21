import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from builder import loader, augment, device
from tools import figure_train_val
from tqdm import tqdm
from prettytable import PrettyTable
import time
from models import get_model
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s [%(levelname)s] %(message)s')


def train_model(train_loader, val_loader, device, model_name, num_epochs=10):
    model = get_model(model_name, num_classes=30)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    acc_train, acc_val = [], []
    loss_train, loss_val = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training")
        for images, labels in train_bar:
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

        print(f"Epoch {epoch + 1}/{num_epochs} Training - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation")
        with torch.no_grad():
            for images, labels in val_bar:
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

        print(f"Epoch {epoch + 1}/{num_epochs} Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

        table = PrettyTable()
        table.field_names = ["Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"]
        table.add_row([epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy])
        print(table)

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
    parser.add_argument('--model_name', type=str, default='resnet18', help="Name of the model to use")

    args = parser.parse_args()

    logger.info("Using model: {}".format(args.model_name))

    train_transform, val_transform = augment()
    train_loader, val_loader = loader(args.train_csv, args.val_csv, args.train_dir, args.val_dir, args.batch_size,
                                      args.num_workers, train_transform, val_transform)
    device = device()

    train_model(train_loader, val_loader, device, args.model_name, args.num_epochs)
