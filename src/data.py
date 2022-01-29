import pandas as pd
import numpy as np
import requests
import zipfile
import torch
from os import path, mkdir
from torch.utils.data import DataLoader, Dataset
from torch import tensor

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

def get_movielens_ratings():
    download_movielens_dataset()  # Download data from web

    # Load the ratings data
    ratings = pd.read_csv(
        path.join('data', 'movie_lens', 'ml-1m', 'ratings.dat'), 
        delimiter='::',
        header=None,
        names=['user_id','item_id','rating','timestamp'],
        engine='python')
    
    return ratings


def train_valid_test_split(df):
    # Split to train and test by time
    first_test_ts = df['timestamp'].quantile(0.98)
    df_train_tmp = df.loc[df['timestamp'] < first_test_ts]
    df_test = df.loc[df['timestamp'] >= first_test_ts]
    # Split the remaining train to train and valid by time
    first_valid_ts = df_train_tmp['timestamp'].quantile(0.98)
    df_train = df_train_tmp.loc[df_train_tmp['timestamp'] < first_valid_ts]
    df_valid = df_train_tmp.loc[df_train_tmp['timestamp'] >= first_valid_ts]
    del df_train_tmp, df
    return df_train, df_valid, df_test


def train_valid_test_split_autorec(df):
    # Split to train and test by time
    first_test_ts = df['timestamp'].quantile(0.98)
    df_train = df.loc[df['timestamp'] < first_test_ts]
    return df_train, df.copy()


def remove_non_trained_users_items(df_train, df_valid, df_test):
    # Remove USERS in valid and test sets that aren't present in the training set
    train_unique_users = df_train['user_id'].unique()
    valid_unique_users = df_valid['user_id'].unique()
    test_unique_users = df_test['user_id'].unique()
    # Clean valid set
    valid_missing_in_train = set(valid_unique_users).difference(set(train_unique_users))
    df_valid = df_valid.loc[~df_valid['user_id'].isin(valid_missing_in_train)]
    # Clean test set
    test_missing_in_train = set(test_unique_users).difference(set(train_unique_users))
    df_test = df_test.loc[~df_test['user_id'].isin(test_missing_in_train)]
    # Validations
    assert(df_train['user_id'].nunique() >= df_valid['user_id'].nunique())
    assert(df_train['user_id'].nunique() >= df_test['user_id'].nunique())

    # Remove ITEMS in valid and test sets that aren't present in the training set
    train_unique_items = df_train['item_id'].unique()
    valid_unique_items = df_valid['item_id'].unique()
    test_unique_items = df_test['item_id'].unique()
    # Clean valid set
    valid_missing_in_train = set(valid_unique_items).difference(set(train_unique_items))
    df_valid = df_valid.loc[~df_valid['item_id'].isin(valid_missing_in_train)]
    # Clean test set
    test_missing_in_train = set(test_unique_items).difference(set(train_unique_items))
    df_test = df_test.loc[~df_test['item_id'].isin(test_missing_in_train)]
    # Validations
    assert(df_train['item_id'].nunique() >= df_valid['item_id'].nunique())
    assert(df_train['item_id'].nunique() >= df_test['item_id'].nunique())
    
    return df_train, df_valid, df_test


def create_ratings_matrix(ratings, df_train, df_valid, df_test):
    ratings_pivot = ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    train_pivot = df_train.pivot(index = 'user_id', columns ='item_id', values = 'rating').fillna(0)
    valid_pivot = df_valid.pivot(index = 'user_id', columns ='item_id', values = 'rating').fillna(0)
    test_pivot = df_test.pivot(index = 'user_id', columns ='item_id', values = 'rating').fillna(0)
    del ratings, df_train, df_valid, df_test

    ratings_train = pd.DataFrame(np.zeros(ratings_pivot.shape), index=ratings_pivot.index, columns=ratings_pivot.columns)
    ratings_valid = ratings_train.copy()
    ratings_test = ratings_train.copy()

    ratings_train.loc[train_pivot.index, train_pivot.columns] = train_pivot.values
    ratings_valid.loc[valid_pivot.index, valid_pivot.columns] = valid_pivot.values
    ratings_test.loc[test_pivot.index, test_pivot.columns] = test_pivot.values

    return ratings_train, ratings_valid, ratings_test


class VectorsDataSet(Dataset):
    def __init__(self, ratings_matrix, by_user=True) -> None:
        self.data = tensor(ratings_matrix.values).float()
        self.by_user = by_user

    def __getitem__(self, index: int):
        if self.by_user:
            vec = self.data[index]
        else:
            vec = self.data[:, index]
        return vec

    def __len__(self) -> int:
        if self.by_user:
            return self.data.shape[0]
        else:
            return self.data.shape[1]


def movielens_dataloaders(by_user:bool = True, batch_size:int = 128):
    """
    """
    ratings = get_movielens_ratings()

    df_train_tmp, df_test = train_valid_test_split_autorec(df=ratings)
    df_train, df_valid = train_valid_test_split_autorec(df=df_train_tmp)

    # df_train, df_valid, df_test = train_valid_test_split(df=ratings)
    df_train, df_valid, df_test = remove_non_trained_users_items(df_train, df_valid, df_test)
    ratings_train, ratings_valid, ratings_test = create_ratings_matrix(ratings, df_train, df_valid, df_test)

    ds_train = VectorsDataSet(ratings_matrix=ratings_train, by_user=by_user)
    ds_valid = VectorsDataSet(ratings_matrix=ratings_valid, by_user=by_user)
    ds_test = VectorsDataSet(ratings_matrix=ratings_test, by_user=by_user)

    dl_train = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = DataLoader(dataset=ds_valid, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(dataset=ds_test, batch_size=batch_size, shuffle=True)

    return dl_train, dl_valid, dl_test


class RatingDataset(Dataset):
    def __init__(self, df):
        self.num_samples = len(df)
        self.users = tensor(df['user_id'].values)
        self.items = tensor(df['item_id'].values)
        self.labels = tensor(df['rating'].values).float()
        self.num_users = df['user_id'].max()
        self.num_items = df['item_id'].max()

    def __getitem__(self, index):
        user = self.users[index]
        item = self.items[index]
        label = self.labels[index].item()
        return user, item, label

    def __len__(self):
        return self.num_samples


def movielens_mf_dataloaders(batch_size:int = 128):
    """
    """
    ratings = get_movielens_ratings()
    df_train, df_valid, df_test = train_valid_test_split(df=ratings)
    df_train, df_valid, df_test = remove_non_trained_users_items(df_train, df_valid, df_test)

    ds_train = RatingDataset(df=df_train)
    ds_valid = RatingDataset(df=df_valid)
    ds_test = RatingDataset(df=df_test)

    dl_train = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = DataLoader(dataset=ds_valid, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(dataset=ds_test, batch_size=batch_size, shuffle=True)

    return dl_train, dl_valid, dl_test