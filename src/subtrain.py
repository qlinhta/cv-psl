import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from builder import loader, augment, device
from tools import figure_train_val
from tqdm import tqdm
import logging
import coloredlogs
from models import get_model_by_id

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s [%(levelname)s] %(message)s')


def save_model(model, epoch, optimizer, path):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, path)


def train_model(train_loader, val_loader, device, model_id, num_epochs, num_classes):
    model_info = get_model_by_id(model_id)
    model = model_info.get_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training")

        for batch_idx, (images, labels) in enumerate(train_bar):
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

            train_bar.set_postfix(loss=running_loss / (batch_idx + 1), accuracy=100 * correct / total)

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation")

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_bar):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                val_bar.set_postfix(loss=val_loss / (batch_idx + 1), accuracy=100 * correct / total)

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        save_model(model, epoch + 1, optimizer, f"checkpoint_epoch_{epoch + 1}.pth")


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
    train_model(train_loader, val_loader, device, args.model_id, args.num_epochs, args.num_classes)
