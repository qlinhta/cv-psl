import argparse
import os
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from models import get_model_by_id
from builder import augment, device
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s [%(levelname)s] %(message)s')


def load_model(model_id, model_path, device):
    model_info = get_model_by_id(model_id)
    model = model_info.get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def predict(model, device, test_dir, output_csv, transform):
    image_files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]

    predictions = []

    for image_file in tqdm(image_files, desc="Predicting"):
        image_path = os.path.join(test_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            predictions.append({'ID': image_file.split('.')[0], 'Category': predicted.item()})

    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)
    logger.info(f"Predictions saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for bird classification model")
    parser.add_argument('--test_dir', type=str, required=True, help="Path to the test images directory")
    parser.add_argument('--model_id', type=int, required=True, help="ID of the model to use")
    parser.add_argument('--num_classes', type=int, default=30, help="Number of classes in the model")

    args = parser.parse_args()

    device = device()
    logger.info(f"Using device: {device}")

    model_info = get_model_by_id(args.model_id)
    model_name = model_info.get_name()
    model_path = f'saved_models/{model_name}_best.pth'
    logger.info(f"Model path: {model_path}")

    _, val_transform = augment()

    model = load_model(args.model_id, model_path, device)

    output_csv = f'./submissions/submission_{model_name}.csv'
    predict(model, device, args.test_dir, output_csv, val_transform)
