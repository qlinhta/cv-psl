import os
import requests
import zipfile
from io import BytesIO
import subprocess
import sys


# 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


def setup_git_config(email, name):
    print("Configuring git user email and name...")
    subprocess.run(['git', 'config', '--global', 'user.email', email], check=True)
    subprocess.run(['git', 'config', '--global', 'user.name', name], check=True)
    print("Git configuration completed.")


def main():
    with zipfile.ZipFile('raw-dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('.')

    email = "qlinhta@outlook.com"
    name = "Quyen Linh TA"
    setup_git_config(email, name)


if __name__ == "__main__":
    main()
