import pandas as pd
import optuna
import torch
import torch.optim as optim

import src.AutoRec_trainer as autorec_trainer
import src.MF_trainer as mf_trainer
import src.VAE_trainer as vae_trainer
from src.models import get_model
from src.data import get_data
from src.loss import mrr


def train(model_name, model, optimizer, epochs, dl_train, dl_test, device, dataset_name):
    """
    Execute the proper trainer with the right model, optimizer and relevant data loaders.
    """
    loss = None
    if model_name == 'MF':
        loss = mf_trainer.train_mf(
            model=model, optimizer=optimizer, max_epochs=epochs, early_stopping=3, 
            dl_train=dl_train, dl_test=dl_test, device=device, dataset_name=dataset_name)
    elif model_name == 'AutoRec':
        loss = autorec_trainer.train_autorec(
            model=model, optimizer=optimizer, max_epochs=epochs, early_stopping=3, 
            dl_train=dl_train, dl_test=dl_test, device=device, dataset_name=dataset_name)
    elif model_name == 'VAE':
        loss = vae_trainer.train_vae(
            model=model, optimizer=optimizer, max_epochs=epochs, early_stopping=3, 
            dl_train=dl_train, dl_test=dl_test, device=device, dataset_name=dataset_name)
    return loss, model

def tune_params(model_name, dataset_name, n_trials, max_epochs, device):
    """
    Use the Optuna package for hyperparameters tuning.
    - Define the ranges for the relevant hyperparameters
    - Sample different hyperparameters combinations using RandomSampler (can be changed)
    - Train the model using the sample and keep the validation loss.
    After many trials, decide on the best hyperparameters and return both the trials results and the entire study.
    """
    
    def objective(trial):
        # Define all hyperparams range options (for different datasets and models)
        params_dict = {
            'movielens': {
                'MF': {
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
                    'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
                    'latent_dim': trial.suggest_int("latent_dim", 10, 20),
                    'batch_size': trial.suggest_categorical("batch_size", [128, 256, 512])
                },
                'AutoRec': {
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
                    'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
                    'latent_dim': trial.suggest_int("latent_dim", 10, 20),
                    'batch_size': trial.suggest_categorical("batch_size", [128, 256, 512])
                },
                'VAE': {
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
                    'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
                    'batch_size': trial.suggest_categorical("batch_size", [128, 256, 512])
                }
            },
            'netflix': {
                'MF': {
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
                    'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
                    'latent_dim': trial.suggest_int("latent_dim", 10, 20),
                    'batch_size': trial.suggest_categorical("batch_size", [128, 256, 512])
                },
                'AutoRec': {
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
                    'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
                    'latent_dim': trial.suggest_int("latent_dim", 10, 20),
                    'batch_size': trial.suggest_categorical("batch_size", [128, 256, 512])
                },
                'VAE': {
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
                    'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
                    'batch_size': trial.suggest_categorical("batch_size", [128, 256, 512])
                }
            }
        }
        
        params = params_dict[dataset_name][model_name]  # Get the relevant params range by the dataset and model
        dl_train, dl_valid, _, _ = get_data(model_name=model_name, dataset_name=dataset_name, batch_size=params['batch_size'])
        model = get_model(model_name, params, dl_train)  # Build model
        optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr= params['learning_rate'])  # Instantiate optimizer
        valid_loss, _ = train(model_name, model, optimizer, max_epochs, dl_train, dl_valid, device, dataset_name)  # Train the model and calc the validation loss

        return valid_loss


    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler())   # Build the study
    study.optimize(objective, n_trials=n_trials)   # Optimize (the actual tuning process)
    df_trials_results = study.trials_dataframe()    # Extract the trials information as Pandas DataFrame

    return study, df_trials_results


def calc_final_mrr(model_name, model, dl_test):
    """
    Calculate the final MRR score for the fully trained model.
    Based on the model type, extract the entire test set, 
    evaluate it with the trained model and calculate the MRR score.
    """
    if model_name == 'MF':
        test_users, test_items, y_test = dl_test.dataset.get_all_data()

        # Generate predictions
        model.eval()
        with torch.no_grad():
            y_preds = model(test_users, test_items)

        # Prepare full rating matrixs for the MRR calculation
        # True rating matrix
        df_true = pd.DataFrame({'user_id': test_users, 'item_id': test_items, 'rating': y_test})
        ratings_true = df_true.pivot(index = 'user_id', columns ='item_id', values = 'rating').fillna(0)
        # Predicted rating matrix
        df_preds = pd.DataFrame({'user_id': test_users, 'item_id': test_items, 'rating': y_preds})
        ratings_preds = df_preds.pivot(index = 'user_id', columns ='item_id', values = 'rating').fillna(0)
        mrr_ = mrr(pred=ratings_preds.values, actual=ratings_true.values, cutoff=5, mrr_threshold=4)
    else:  # AutoRec & VAE
        x = y_test = dl_test.dataset.get_all_data()
        # Generate predictions
        model.eval()
        with torch.no_grad():
            y_preds = model(x)

        # In AutoRec & VAE the predictions are already in the shape of rating matrixs so no need in processing
        mrr_ = mrr(pred=y_preds.numpy(), actual=y_test.numpy(), cutoff=5, mrr_threshold=4)
    
    return mrr_


def final_train(model_name, dataset_name, best_params, max_epochs, dl_full_train, dl_test, device):
    """
    After we optimized and choosed the best hyperparameters for the model we want to prepare it for predicting the test set.
    - Use the best hyperparameters to build the final model
    - Train the final model on the train+validation data sets (full_train)
    - Test it against the test set for final results
    """
    _, _, dl_test, dl_full_train = get_data(model_name=model_name, dataset_name=dataset_name, batch_size=best_params['batch_size'])
    model = get_model(model_name, best_params, dl_full_train)  # Build model
    optimizer = getattr(optim, best_params['optimizer'])(model.parameters(), lr= best_params['learning_rate'])  # Instantiate optimizer
    # Train the model on the full_train (train+valid) set and calc the test loss
    test_loss, final_model = train(model_name, model, optimizer, max_epochs, dl_full_train, dl_test, device, dataset_name)
    
    mrr_ = calc_final_mrr(model_name=model_name, model=final_model, dl_test=dl_test)

    return test_loss, final_model, mrr_


#### Only for testing
if __name__ == '__main__':
    import torch
    from src.hyperparams_tuning import *
    from src.data import *

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = 'movielens'
    best_params = {'batch_size': 128, 'learning_rate': 0.001, 'optimizer': 'Adam'}
    max_epochs = 2
    model_name = 'MF' #'AutoRec' # 'MF'

    dl_train, dl_valid, dl_test, dl_full_train = get_data(model_name=model_name, dataset_name=dataset_name, batch_size=best_params['batch_size'])
    model = get_model(model_name, best_params, dl_train)  # Build model
    optimizer = getattr(optim, best_params['optimizer'])(model.parameters(), lr= best_params['learning_rate'])  # Instantiate optimizer
    test_loss, final_model, mrr_ = final_train(model_name, dataset_name, best_params, max_epochs, device)
    print(test_loss)
    print(final_model)
    print(mrr_)