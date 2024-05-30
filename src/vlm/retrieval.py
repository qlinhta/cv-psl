"""import sys
import os
import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from rich.logging import RichHandler
from rich.progress import track
import logging
import argparse
from concurrent.futures import ProcessPoolExecutor
import re

logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("rich")

device = torch.device('cpu')

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)


def extract_bird_label(filename):
    match = re.match(r'([a-zA-Z_]+)', filename)
    if match:
        bird_label = match.group(0).replace('_', ' ')
        return bird_label
    return "a bird"


def generate_caption(image_path):
    image = Image.open(image_path).convert('RGB')
    bird_label = extract_bird_label(os.path.basename(image_path))
    prompt_prefix = f"this is a {bird_label}."

    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs['pixel_values']

    out = model.generate(pixel_values=pixel_values, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)

    full_caption = prompt_prefix + caption
    return os.path.splitext(os.path.basename(image_path))[0], full_caption


def process_image(args):
    image_dir, img_name = args
    image_path = os.path.join(image_dir, img_name)
    return generate_caption(image_path)


def generate_prompts(image_dir, output_file, num_workers=8):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.write('image,prompt\n')

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for img_name, caption in track(executor.map(process_image, [(image_dir, img_name) for img_name in image_files]),
                                       total=len(image_files), description="Processing..."):
            with open(output_file, 'a') as f:
                f.write(f"{img_name},{caption}\n")
                f.flush()

    logger.info(f"Prompts saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate prompts for images.')
    parser.add_argument('--train_dir', required=True, help='Directory containing training images')
    parser.add_argument('--val_dir', required=True, help='Directory containing validation images')
    args = parser.parse_args()

    train_output = os.path.join('./dataset/', 'train_prompts.csv')
    val_output = os.path.join('./dataset/', 'val_prompts.csv')

    generate_prompts(args.train_dir, train_output)
    generate_prompts(args.val_dir, val_output)


if __name__ == '__main__':
    main()
"""

import sys
import os
import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from rich.logging import RichHandler
from rich.progress import track
import logging
import argparse
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("rich")

device = torch.device('cpu')

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)


def generate_caption(image_path):
    image = Image.open(image_path).convert('RGB')

    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs['pixel_values']

    out = model.generate(pixel_values=pixel_values, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return os.path.splitext(os.path.basename(image_path))[0], caption


def process_image(args):
    image_dir, img_name = args
    image_path = os.path.join(image_dir, img_name)
    return generate_caption(image_path)


def generate_prompts(image_dir, output_file, num_workers=8):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.write('image,prompt\n')

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for img_name, caption in track(executor.map(process_image, [(image_dir, img_name) for img_name in image_files]),
                                       total=len(image_files), description="Processing..."):
            with open(output_file, 'a') as f:
                f.write(f"{img_name},{caption}\n")
                f.flush()

    logger.info(f"Prompts saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate prompts for images.')
    parser.add_argument('--train_dir', required=False, help='Directory containing training images')
    parser.add_argument('--val_dir', required=False, help='Directory containing validation images')
    parser.add_argument('--test_dir', required=False, help='Directory containing test images')
    args = parser.parse_args()

    if args.train_dir:
        train_output = os.path.join('./dataset/', 'train_prompts.csv')
        generate_prompts(args.train_dir, train_output)

    if args.val_dir:
        val_output = os.path.join('./dataset/', 'val_prompts.csv')
        generate_prompts(args.val_dir, val_output)

    if args.test_dir:
        test_output = os.path.join('./dataset/', 'test_prompts.csv')
        generate_prompts(args.test_dir, test_output)


if __name__ == '__main__':
    main()
