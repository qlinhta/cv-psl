import os
import requests
import zipfile
from io import BytesIO
import subprocess


def download_and_extract_zip_from_google_drive(file_id, extract_to='./raw-dataset'):
    google_drive_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(google_drive_url)

    if 'content-disposition' not in response.headers:
        warning_url = google_drive_url + "&confirm=t"
        response = requests.get(warning_url)

    with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
        print(f'Extracting files to {extract_to}')
        zip_file.extractall(extract_to)
    print("Extraction complete!")


def setup_git_config(email, name):
    print("Configuring git user email and name...")
    subprocess.run(['git', 'config', '--global', 'user.email', email], check=True)
    subprocess.run(['git', 'config', '--global', 'user.name', name], check=True)
    print("Git configuration completed.")


def main():
    file_id = "1IEnpbGjNqXYF4vPY1NW-ODcrYZomyb4S"
    raw_dataset_path = './raw-dataset'

    if not os.path.exists(raw_dataset_path):
        os.makedirs(raw_dataset_path)

    download_and_extract_zip_from_google_drive(file_id, raw_dataset_path)

    email = "qlinhta@outlook.com"
    name = "Quyen Linh TA"
    setup_git_config(email, name)


if __name__ == "__main__":
    main()
