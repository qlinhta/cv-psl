import pandas as pd
import os


def clean_csv(csv_path, cleaned_csv_path):
    df = pd.read_csv(csv_path)
    df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: x.encode('utf-8').decode('utf-8').strip())
    df.to_csv(cleaned_csv_path, index=False)
    print(f"Cleaned CSV saved to {cleaned_csv_path}")


if __name__ == "__main__":
    train_csv = './dataset/train_labels.csv'
    val_csv = './dataset/val_labels.csv'
    clean_train_csv = './dataset/train_labels_utf8.csv'
    clean_val_csv = './dataset/val_labels_utf8.csv'

    clean_csv(train_csv, clean_train_csv)
    clean_csv(val_csv, clean_val_csv)
