import os
import subprocess
import zipfile


def download_dataset_from_google_drive(file_id, output_path):
    subprocess.run(['gdown', f'https://drive.google.com/uc?id={file_id}', '-O', 'dataset.zip'])
    if not zipfile.is_zipfile('dataset.zip'):
        raise ValueError("Downloaded file is not a valid zip file")
    with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
        zip_ref.extractall(output_path)
    os.remove('dataset.zip')


def main():
    google_drive_file_id = '1IEnpbGjNqXYF4vPY1NW-ODcrYZomyb4S'
    raw_dataset_path = os.path.join(os.getcwd(), 'raw-dataset')

    if not os.path.exists(raw_dataset_path):
        os.makedirs(raw_dataset_path)

    download_dataset_from_google_drive(google_drive_file_id, raw_dataset_path)


if __name__ == "__main__":
    main()
