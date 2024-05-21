import os
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTextModel, CLIPTokenizer
import torch
import logging
import coloredlogs
from tqdm import tqdm
import argparse
from builder import device

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s [%(levelname)s] %(message)s')

device = device()

processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
text_model = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32').to(device)

prompts = [
    "a small yellow bird with a black head",
    "a large brown bird with a long beak",
    "a bird with blue feathers and a white belly",
    "a bird with red and green plumage",
    "a black bird with orange wings",
    "a bird with a long tail and spotted wings",
    "a bird with a curved beak and vibrant colors",
    "a bird with a short beak and patterned feathers",
    "a bird perched on a branch in a forest",
    "a bird flying over a lake",
    "a bird with a crown of feathers",
    "a bird with a distinctive song",
    "a bird building a nest",
    "a bird hunting for insects",
    "a bird with a unique crest",
    "a bird with a sleek body and sharp talons",
    "a bird with a colorful beak",
    "a bird with intricate feather patterns",
    "a bird with a white and black striped tail",
    "a bird with a red breast and short wings",
    "a bird with vibrant green and yellow feathers",
    "a bird with a long neck and elegant posture",
    "a bird with bright red and blue plumage",
    "a bird with a short, stout body and brown feathers",
    "a bird with black and white markings and a long beak",
    "a bird with yellow and black striped wings",
    "a bird with a white head and a dark body",
    "a bird with colorful feathers and a loud call",
    "a bird with long legs and a slender body",
    "a bird with a sharp, curved beak and red eyes",
    "a bird with fluffy feathers and a playful demeanor"
]


def generate_prompt(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
    image_features = model.get_image_features(**inputs)

    text_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    text_features = text_model(**text_inputs).last_hidden_state.mean(dim=1)

    similarities = torch.matmul(image_features, text_features.T).squeeze()

    if similarities.dim() == 0:
        similarities = similarities.unsqueeze(0)

    best_prompt_idx = torch.argmax(similarities).item()

    return prompts[best_prompt_idx]


def generate_prompts(image_dir, output_file):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    prompt_list = []

    logger.info(f"Generating prompts for images in {image_dir}")
    for img_name in tqdm(image_files):
        image_path = os.path.join(image_dir, img_name)
        prompt = generate_prompt(image_path)
        prompt_list.append([os.path.splitext(img_name)[0], prompt])

    prompt_df = pd.DataFrame(prompt_list, columns=['image', 'prompt'])
    prompt_df.to_csv(output_file, index=False)
    logger.info(f"Prompts saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate prompts for CUB dataset images.')
    parser.add_argument('--train_dir', required=True, help='Directory containing training images')
    parser.add_argument('--val_dir', required=True, help='Directory containing validation images')
    parser.add_argument('--train_output', required=True, help='Path to save training prompts CSV file')
    parser.add_argument('--val_output', required=True, help='Path to save validation prompts CSV file')
    parser.add_argument('--test_dir', help='Directory containing test images (optional)')
    parser.add_argument('--test_output', help='Path to save test prompts CSV file (optional)')

    args = parser.parse_args()

    generate_prompts(args.train_dir, args.train_output)
    generate_prompts(args.val_dir, args.val_output)

    if args.test_dir and args.test_output:
        generate_prompts(args.test_dir, args.test_output)


if __name__ == '__main__':
    main()
