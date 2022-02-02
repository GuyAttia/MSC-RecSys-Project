import requests
import tarfile
import zipfile
import pandas as pd
import gc
from os import path, mkdir, listdir
from tqdm import tqdm


def create_data_dir(dataset_name:str) -> str:
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

    ## Download only if file doesn't exists
    if not path.isfile(zip_name):
        print('Downloading the MovieLens data')
        r = requests.get(url, allow_redirects=True)
        open(zip_name, 'wb').write(r.content)
        # Extract the zip file
        with zipfile.ZipFile(zip_name, "r") as zip_ref:
            zip_ref.extractall(path=data_dir)
    else:
        print('Movie-lens file already exists')

### ----------------------------------------------------------------------- Netflix -------------------------------------------------------------------------------- ###

def download_netflix_dataset():
    """
    Download the Netflix-Prize dataset from the web
    """
    data_dir = create_data_dir(dataset_name='netflix')
    
    url = 'https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz'
    zip_name = path.join(data_dir, url.split('/')[-1])

    ## Download only if file doesn't exists
    if not path.isfile(zip_name):
        print('Downloading the Netflix-Prize data - it make take time - please wait')
        r = requests.get(url, allow_redirects=True)
        open(zip_name, 'wb').write(r.content)
    else:
        print('Netflix-Prize file already exists')

    # Extract the tar file
    if not path.isdir(path.join(data_dir, 'download')):
        with tarfile.open(zip_name) as tar_:
            tar_.extractall(data_dir)

    # Etract the nested rating tar file
    nested_zip_name = path.join(data_dir, 'download', 'training_set.tar')
    if not path.isdir(path.join(data_dir, 'training_set')):
        with tarfile.open(nested_zip_name) as tar_:
            tar_.extractall(data_dir)


def netflix_build_rating_table():
    """
    Iterate over the movies rating txt files and create one ratings DataFrame from them
    """
    if path.isfile(path.join('data', 'netflix', 'ratings.csv')):
        print('Netflix ratings csv is already exists')
        df = pd.read_csv(path.join('data', 'netflix', 'ratings.csv'))
    else:
        print('Start processing the Netflix data')
        # Load the rating files
        users = []
        items = []
        ratings = []
        timestamps = []

        num_of_movies = 100
        files_path = path.join('data', 'netflix', 'training_set')
        for i, file_name_ in tqdm(enumerate(listdir(files_path))):
            if i == num_of_movies:
                break
            file_name = path.join(files_path, file_name_)
            with open(file_name, "r") as f:
                movie = -1
                for line in f:
                    if line.endswith(':\n'):
                        movie = int(line[:-2]) - 1
                        continue
                    assert movie >= 0
                    splitted = line.split(',')
                    user = int(splitted[0])
                    rating = float(splitted[1])
                    timestamp_ = splitted[2]
                    users.append(user)
                    items.append(movie)
                    ratings.append(rating)
                    timestamps.append(timestamp_[:-1])
            gc.collect()
        df = pd.DataFrame({'user_id': users, 'item_id': items, 'rating': ratings, 'timestamp': timestamps})
        df.to_csv(path.join('data', 'netflix', 'ratings.csv'), index=False)

    return df



if __name__ == '__main__':
    # download_netflix_dataset()
    netflix_build_rating_table()