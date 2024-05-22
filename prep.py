import os
import subprocess


def setup_git_config(username, email):
    subprocess.run(['git', 'config', '--global', 'user.name', username])
    subprocess.run(['git', 'config', '--global', 'user.email', email])


def clone_private_repo(repo_url, token):
    repo_url_with_token = repo_url.replace('https://', f'https://{token}@')
    subprocess.run(['git', 'clone', repo_url_with_token])


def download_kaggle_competition_data(competition, download_path):
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    subprocess.run(['kaggle', 'competitions', 'download', '-c', competition, '--path', download_path])


def main():
    repo_url = "https://github.com/qlinhta/cv-psl.git"
    token = "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"
    kaggle_competition = 'm-2-iasd-app-dlia-project-2024'
    raw_dataset_path = os.path.join(os.getcwd(), 'cv-psl', 'raw-dataset')
    github_user = "Quyen Linh TA"
    github_email = "qlinhta@outlook.com"

    setup_git_config(github_user, github_email)
    clone_private_repo(repo_url, token)
    download_kaggle_competition_data(kaggle_competition, raw_dataset_path)


if __name__ == "__main__":
    main()
