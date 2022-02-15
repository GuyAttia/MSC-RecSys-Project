import zipfile
from os import mkdir, path

import requests


def create_data_dir(dataset_name: str) -> str:
    """
    Check if the data directory exists and create it if not

    :param dataset_name: The name of the downloaded dataset
    :return: The full path of the created directory
    """
    if not path.isdir('data'):
        mkdir('data')

    data_dir = path.join('data', dataset_name)
    if not path.isdir(data_dir):
        mkdir(data_dir)
    return data_dir

### ----------------------------------------------------------------------- Movielens 1M -------------------------------------------------------------------------------- ###


def download_movielens_dataset():
    """
    Download the movielens 1M dataset from the web
    """
    data_dir = create_data_dir(dataset_name='movie_lens')

    url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
    zip_name = path.join(data_dir, url.split('/')[-1])

    # Download only if file doesn't exists
    if not path.isfile(zip_name):
        print('Downloading the MovieLens data')
        r = requests.get(url, allow_redirects=True)
        open(zip_name, 'wb').write(r.content)
        # Extract the zip file
        with zipfile.ZipFile(zip_name, "r") as zip_ref:
            zip_ref.extractall(path=data_dir)
    else:
        print('Movie-lens file already exists')

### ----------------------------------------------------------------------- Books -------------------------------------------------------------------------------- ###


def download_books_dataset():
    """
    Download the Books (BX) dataset from the web
    """
    data_dir = create_data_dir(dataset_name='books')

    url = 'http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip'
    zip_name = path.join(data_dir, url.split('/')[-1])

    # Download only if file doesn't exists
    if not path.isfile(zip_name):
        print('Downloading the Books data - it make take time - please wait')
        r = requests.get(url, allow_redirects=True)
        open(zip_name, 'wb').write(r.content)
        # Extract the zip file
        with zipfile.ZipFile(zip_name, "r") as zip_ref:
            zip_ref.extractall(path=data_dir)
    else:
        print('Books file already exists')


if __name__ == '__main__':
    download_books_dataset()
