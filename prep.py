import os
import subprocess
import zipfile


def download_dataset_from_google_drive(url, output_path):
    subprocess.run(['curl', '-L', '-s', '-o', 'dataset.zip', url])
    with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
        zip_ref.extractall(output_path)
    os.remove('dataset.zip')


def main():
    google_drive_url = 'https://drive.google.com/uc?id=1IEnpbGjNqXYF4vPY1NW-ODcrYZomyb4S&confirm=t'
    raw_dataset_path = os.path.join(os.getcwd(), 'raw-dataset')

    if not os.path.exists(raw_dataset_path):
        os.makedirs(raw_dataset_path)

    download_dataset_from_google_drive(google_drive_url, raw_dataset_path)


if __name__ == "__main__":
    main()
