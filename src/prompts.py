import argparse
import pandas as pd
import os


def generate_prompts(data_path, output_folder, output_filename):
    df = pd.read_csv(data_path)
    df['prompt'] = df['label'].apply(lambda x: f"This is a {' '.join(x.split('.')[1].split('_'))} bird")
    output_path = os.path.join(output_folder, output_filename)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True, help='Path to train CSV file')
    parser.add_argument('--val_csv', required=True, help='Path to validation CSV file')
    parser.add_argument('--output_folder', required=True, help='Path to output folder for prompts CSV files')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    generate_prompts(args.train_csv, args.output_folder, 'train_prompts.csv')
    generate_prompts(args.val_csv, args.output_folder, 'val_prompts.csv')
