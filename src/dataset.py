import os
import shutil
import pandas as pd
import argparse
from tqdm import tqdm


def dataset(raw_dataset_path, output_dataset_path, classes_file):
    if not os.path.exists(output_dataset_path):
        os.makedirs(output_dataset_path)

    train_output_path = os.path.join(output_dataset_path, 'train')
    val_output_path = os.path.join(output_dataset_path, 'val')
    test_output_path = os.path.join(output_dataset_path, 'test')

    for path in [train_output_path, val_output_path, test_output_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    classes_df = pd.read_csv(classes_file)
    category_to_idx = dict(zip(classes_df['category_cub'], classes_df['idx']))

    train_labels = []
    val_labels = []

    for split in ['train', 'val']:
        split_images_path = os.path.join(raw_dataset_path, f'{split}_images')
        categories = os.listdir(split_images_path)
        for category in tqdm(categories, desc=f'Processing {split} set'):
            category_path = os.path.join(split_images_path, category)
            if os.path.isdir(category_path):
                category_idx = category_to_idx.get(category, None)
                if category_idx is not None:
                    images = os.listdir(category_path)
                    for img_name in images:
                        img_name_wo_ext = os.path.splitext(img_name)[0]
                        src_img_path = os.path.join(category_path, img_name)
                        if split == 'train':
                            dst_img_path = os.path.join(train_output_path, img_name)
                            train_labels.append([img_name_wo_ext, category_idx, category])
                        else:
                            dst_img_path = os.path.join(val_output_path, img_name)
                            val_labels.append([img_name_wo_ext, category_idx, category])
                        shutil.copyfile(src_img_path, dst_img_path)

    train_labels_df = pd.DataFrame(train_labels, columns=['image_name', 'label_idx', 'label'])
    val_labels_df = pd.DataFrame(val_labels, columns=['image_name', 'label_idx', 'label'])

    train_labels_df.to_csv(os.path.join(output_dataset_path, 'train.csv'), index=False)
    val_labels_df.to_csv(os.path.join(output_dataset_path, 'val.csv'), index=False)

    test_images_path = os.path.join(raw_dataset_path, 'test_images')
    test_images = os.listdir(test_images_path)
    for img_name in tqdm(test_images, desc='Processing test set'):
        src_img_path = os.path.join(test_images_path, img_name)
        dst_img_path = os.path.join(test_output_path, img_name)
        shutil.copyfile(src_img_path, dst_img_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for bird classification")
    parser.add_argument('--raw_dataset', type=str, required=True, help="Path to the raw dataset directory")
    parser.add_argument('--output_dataset', type=str, required=True, help="Path to the output dataset directory")
    parser.add_argument('--classes_file', type=str, required=True, help="Path to the classes indexes CSV file")

    args = parser.parse_args()

    dataset(args.raw_dataset, args.output_dataset, args.classes_file)
