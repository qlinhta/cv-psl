import argparse
import json
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from builder import loader, augment, device
from main import train_model
from models import get_model_by_id
from tqdm import tqdm
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s [%(levelname)s] %(message)s')


def objective(trial, config, train_csv, val_csv, train_dir, val_dir, num_workers, num_epochs, model_id):
    train_transform, val_transform = augment()

    batch_size = trial.suggest_categorical("batch_size", config['batch_size']['values'])
    train_loader, val_loader = loader(
        train_csv, val_csv, train_dir, val_dir, batch_size, num_workers, train_transform, val_transform
    )

    device_used = device()

    model_info = get_model_by_id(model_id)
    model = model_info.get_model().to(device_used)

    lr = trial.suggest_float("lr", config['lr']['min'], config['lr']['max'], log=True)
    weight_decay = trial.suggest_float("weight_decay", config['weight_decay']['min'], config['weight_decay']['max'],
                                       log=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training")
        for images, labels in train_bar:
            images, labels = images.to(device_used), labels.to(device_used)
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

        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation")
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device_used), labels.to(device_used)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_bar.set_postfix(loss=val_running_loss / len(val_loader), accuracy=100 * correct / total)

        val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        logger.info(
            f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

    return best_val_accuracy


def tune_hyperparameters(config, train_csv, val_csv, train_dir, val_dir, num_workers, num_epochs, model_id, n_trials):
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, config, train_csv, val_csv, train_dir, val_dir, num_workers, num_epochs,
                                model_id),
        n_trials=n_trials
    )

    logger.info(f"Best trial: {study.best_trial.value}")
    logger.info(f"Best hyperparameters: {study.best_trial.params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for bird classification model")
    parser.add_argument('--train_csv', type=str, required=True, help="Path to the train labels CSV file")
    parser.add_argument('--val_csv', type=str, required=True, help="Path to the validation labels CSV file")
    parser.add_argument('--train_dir', type=str, required=True, help="Path to the train images directory")
    parser.add_argument('--val_dir', type=str, required=True, help="Path to the validation images directory")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for the dataloaders")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--model_id', type=int, required=True, help="ID of the model to use")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration JSON file")
    parser.add_argument('--fine_tune', action='store_true', help="Flag to fine-tune model hyperparameters")
    parser.add_argument('--n_trials', type=int, default=50, help="Number of trials for hyperparameter tuning")

    args = parser.parse_args()

    if args.fine_tune:
        with open(args.config, 'r') as f:
            config = json.load(f)
        tune_hyperparameters(config, args.train_csv, args.val_csv, args.train_dir, args.val_dir, args.num_workers,
                             args.num_epochs, args.model_id, args.n_trials)
