import pandas as pd
import numpy as np
from os import path
from torch.utils.data import DataLoader, Dataset
from torch import tensor

from src.download_data import download_movielens_dataset

### ----------------------------------------------------------------------- Get Data -------------------------------------------------------------------------------- ###

def get_movielens_ratings():
    """
    Downliad the movielens dataset from the web (if needed) and load the ratings file into a Pandas DataFrame
    """
    # Download data from web
    download_movielens_dataset()  

    # Load the ratings data into a Pandas DataFrame
    ratings = pd.read_csv(
        path.join('data', 'movie_lens', 'ml-1m', 'ratings.dat'), 
        delimiter='::',
        header=None,
        names=['user_id','item_id','rating','timestamp'],
        engine='python')
    
    return ratings

def get_netflix_ratings():
    pass

### ----------------------------------------------------------------------- Split Data -------------------------------------------------------------------------------- ###

def train_valid_test_split(df):
    """
    Split the data into train, validation, test, and full-train sets based on the timestamp.
    """
    # Split to train and test by time
    first_test_ts = df['timestamp'].quantile(0.98)   # Get the 98 quantile of the time to cut the test set
    df_full_train = df.loc[df['timestamp'] < first_test_ts]     # Will be used for training before the test predictions
    df_test = df.loc[df['timestamp'] >= first_test_ts]

    # Split the remaining train to train and valid by time
    first_valid_ts = df_full_train['timestamp'].quantile(0.98)  # Get the 98 quantile of the remaining training time to cut the valid set
    df_train = df_full_train.loc[df_full_train['timestamp'] < first_valid_ts]
    df_valid = df_full_train.loc[df_full_train['timestamp'] >= first_valid_ts]
    del df
    return df_train, df_valid, df_test, df_full_train


def train_valid_test_split_autorec(df):
    """
    Split the data into train and test sets based on the timestamp.
    This method works in a cummulative way, meanning the test set contain all the train does + the additional records the last timestamps has.
    """
    # Split to train and test by time
    first_test_ts = df['timestamp'].quantile(0.98)  # Get the 98 quantile of the time to add only into the test set
    df_train = df.loc[df['timestamp'] < first_test_ts]  # Keep only the first timestamps for training
    return df_train, df.copy()

### ----------------------------------------------------------------------- Processing -------------------------------------------------------------------------------- ###

def remove_non_trained_users_items(df_train, df_valid, df_test, df_full_train=None):
    """
    Make sure non of the users & items appears only on the validation or test sets if we didn't trained on them.
    """
    # Due to the way we split our data for AutoRec and VAE, we can actually use the validation set as our full train set.
    if df_full_train is None:
        df_full_train = df_valid.copy()

    # USERS
    # Remove USERS in valid and test sets that aren't present in the training set
    train_unique_users = df_train['user_id'].unique()
    valid_unique_users = df_valid['user_id'].unique()
    test_unique_users = df_test['user_id'].unique()
    full_train_unique_users = df_full_train['user_id'].unique()
    
    # Clean valid set
    valid_missing_in_train = set(valid_unique_users).difference(set(train_unique_users))
    df_valid = df_valid.loc[~df_valid['user_id'].isin(valid_missing_in_train)]
    # Clean test set
    test_missing_in_train = set(test_unique_users).difference(set(full_train_unique_users))
    df_test = df_test.loc[~df_test['user_id'].isin(test_missing_in_train)]
    # Validations - check we are ok
    assert(df_train['user_id'].nunique() >= df_valid['user_id'].nunique())
    assert(df_full_train['user_id'].nunique() >= df_test['user_id'].nunique())

    # ITEMS
    # Remove ITEMS in valid and test sets that aren't present in the training set
    train_unique_items = df_train['item_id'].unique()
    valid_unique_items = df_valid['item_id'].unique()
    test_unique_items = df_test['item_id'].unique()
    full_train_unique_items = df_full_train['item_id'].unique()
    # Clean valid set
    valid_missing_in_train = set(valid_unique_items).difference(set(train_unique_items))
    df_valid = df_valid.loc[~df_valid['item_id'].isin(valid_missing_in_train)]
    # Clean test set
    test_missing_in_train = set(test_unique_items).difference(set(full_train_unique_items))
    df_test = df_test.loc[~df_test['item_id'].isin(test_missing_in_train)]
    # Validations - check we are ok
    assert(df_train['item_id'].nunique() >= df_valid['item_id'].nunique())
    assert(df_full_train['item_id'].nunique() >= df_test['item_id'].nunique())
    
    return df_train, df_valid, df_test, df_full_train


def create_ratings_matrix(ratings, df_train, df_valid, df_test):
    """
    Change the ratings DataFrames into ratings matrixes.
    This method is used for the AutoRec and VAE where we want to get user / item vector as an input.
    """
    # Pivot the full-data, train, valid and test DataFrames.
    ratings_pivot = ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    train_pivot = df_train.pivot(index = 'user_id', columns ='item_id', values = 'rating').fillna(0)
    valid_pivot = df_valid.pivot(index = 'user_id', columns ='item_id', values = 'rating').fillna(0)
    test_pivot = df_test.pivot(index = 'user_id', columns ='item_id', values = 'rating').fillna(0)
    del ratings, df_train, df_valid, df_test

    # Create empty matrixes in the shape of the full data
    ratings_train = pd.DataFrame(np.zeros(ratings_pivot.shape), index=ratings_pivot.index, columns=ratings_pivot.columns)
    ratings_valid = ratings_train.copy()
    ratings_test = ratings_train.copy()

    # Fill the relevant ratings by the correct rating matrix
    ratings_train.loc[train_pivot.index, train_pivot.columns] = train_pivot.values
    ratings_valid.loc[valid_pivot.index, valid_pivot.columns] = valid_pivot.values
    ratings_test.loc[test_pivot.index, test_pivot.columns] = test_pivot.values

    return ratings_train, ratings_valid, ratings_test

### ----------------------------------------------------------------------- MF -------------------------------------------------------------------------------- ###

class RatingDataset(Dataset):
    """
    Generate rating dataset to use in the MF model, where each sample should be a tuple of (user, item, rating)
    """
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
    Generate the DataLoader objects for MF with the defined batch size
    """
    ratings = get_movielens_ratings()
    df_train, df_valid, df_test, df_full_train = train_valid_test_split(df=ratings)
    df_train, df_valid, df_test, df_full_train = remove_non_trained_users_items(df_train, df_valid, df_test, df_full_train)

    ds_train = RatingDataset(df=df_train)
    ds_valid = RatingDataset(df=df_valid)
    ds_test = RatingDataset(df=df_test)
    ds_full_train = RatingDataset(df=df_full_train)

    dl_train = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = DataLoader(dataset=ds_valid, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(dataset=ds_test, batch_size=batch_size, shuffle=True)
    dl_full_train = DataLoader(dataset=ds_full_train, batch_size=batch_size, shuffle=True)

    return dl_train, dl_valid, dl_test, dl_full_train


def netflix_mf_dataloaders(batch_size:int = 128):
    pass

### ----------------------------------------------------------------------- AutoRec & VAE -------------------------------------------------------------------------------- ###

class VectorsDataSet(Dataset):
    """
    Generate vectors dataset to use in the AutoRec and VAE models, where each sample should be a user / item vector
    """
    def __init__(self, ratings_matrix, by_user=True) -> None:
        self.data = tensor(ratings_matrix.values).float()
        self.by_user = by_user  # If we want we can use this variable to generate DataSets for User / Item AutoRec

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
    Generate the DataLoader objects for AutoRec and VAE models with the defined batch size
    """
    ratings = get_movielens_ratings()

    df_train_tmp, df_test = train_valid_test_split_autorec(df=ratings)
    df_train, df_valid = train_valid_test_split_autorec(df=df_train_tmp)

    df_train, df_valid, df_test, _ = remove_non_trained_users_items(df_train, df_valid, df_test)
    ratings_train, ratings_valid, ratings_test = create_ratings_matrix(ratings, df_train, df_valid, df_test)

    ds_train = VectorsDataSet(ratings_matrix=ratings_train, by_user=by_user)
    ds_valid = VectorsDataSet(ratings_matrix=ratings_valid, by_user=by_user)
    ds_test = VectorsDataSet(ratings_matrix=ratings_test, by_user=by_user)

    dl_train = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = DataLoader(dataset=ds_valid, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(dataset=ds_test, batch_size=batch_size, shuffle=True)
    dl_full_train = DataLoader(dataset=ds_valid, batch_size=batch_size, shuffle=True)

    return dl_train, dl_valid, dl_test, dl_full_train


def netflix_dataloaders(by_user:bool = True, batch_size:int = 128):
    pass