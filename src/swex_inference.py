import argparse
import os
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from swin.swinc import SwinC
from swin.swinc_loader import augment, device
from transformers import CLIPTokenizer
import logging
import coloredlogs
import numpy as np

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s [%(levelname)s] %(message)s')


def load_model(model_id, model_path, device, num_classes):
    model = SwinC(model_id=model_id, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def load_text_prompts(csv_file):
    df = pd.read_csv(csv_file)
    text_prompts = {}
    for index, row in df.iterrows():
        text_prompts[row['image']] = row['prompt']
    return text_prompts


def predict(model, device, test_dir, output_csv, transform, tokenizer, text_prompts):
    image_files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]

    predictions = []

    for image_file in tqdm(image_files, desc="Predicting"):
        image_path = os.path.join(test_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        augmented = transform(image=np.array(image))
        image = augmented['image'].unsqueeze(0).to(device)

        prompt = text_prompts[image_file.split('.')[0]]
        text_inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}  # Move text_inputs to device

        with torch.no_grad():
            output = model(image, text_inputs)
            _, predicted = torch.max(output, 1)
            predictions.append({'ID': image_file.split('.')[0], 'Category': predicted.item()})

    df = pd.DataFrame(predictions)
    df.to_csv(output_csv, index=False)
    logger.info(f"Predictions saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for bird classification model")
    parser.add_argument('--test_dir', type=str, required=True, help="Path to the test images directory")
    parser.add_argument('--test_text', type=str, required=True, help="Path to the test text prompts CSV file")
    parser.add_argument('--model_id', type=int, required=True, help="ID of the model to use")
    parser.add_argument('--num_classes', type=int, default=30, help="Number of classes in the model")

    args = parser.parse_args()

    device = device()
    logger.info(f"Using device: {device}")

    model_path = './saved_models/SwinC_best.pth'
    logger.info(f"Model path: {model_path}")

    _, val_transform = augment()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    model = load_model(args.model_id, model_path, device, args.num_classes)
    text_prompts = load_text_prompts(args.test_text)

    output_csv = f'./submissions/submission_SwinC.csv'
    if not os.path.exists('submissions'):
        os.makedirs('submissions')
    predict(model, device, args.test_dir, output_csv, val_transform, tokenizer, text_prompts)
