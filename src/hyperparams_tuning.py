import optuna
import torch.optim as optim

import src.models as projects_models
import src.trainers as projects_trainers


def tune_params(model_name, dataset_name, n_trials, dl_train, dl_valid, device):
    MAX_EPOCHS = 10
    
    def objective(trial):
        # Define all hyperparams range options (for different datasets and models)
        params_dict = {
            'movielens': {
                'MF': {
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
                    'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
                    'latent_dim': trial.suggest_int("latent_dim", 10, 20)
                },
                'AutoRec': {
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
                    'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
                },
                'VAE': {
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
                    'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
                }
            }
        }
        # Get the relevant params range by the dataset and model
        params = params_dict[dataset_name][model_name]

        if model_name == 'MF':
            num_users = dl_train.dataset.num_users
            num_items = dl_train.dataset.num_items
            model = projects_models.MF(num_users=num_users, num_items=num_items, params=params)
        elif model_name == 'AutoRec':
            n_dim = dl_train.dataset.__getitem__(1).shape[0]
            linear_encoder = projects_models.EncoderLinear(in_dim=n_dim, params=params)
            linear_decoder = projects_models.DecoderLinear(out_dim=n_dim, params=params)
            model = projects_models.AutoRec(encoder=linear_encoder, decoder=linear_decoder)
        elif model_name == 'VAE':
            pass

        optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr= params['learning_rate'])

        if model_name == 'MF':
            val_loss = projects_trainers.train_mf(
                model=model, optimizer=optimizer, max_epochs=MAX_EPOCHS, early_stopping=3, 
                dl_train=dl_train, dl_test=dl_valid, device=device, dataset_name=dataset_name)
        elif model_name == 'AutoRec':
            val_loss = projects_trainers.train_autorec(
                model=model, optimizer=optimizer, max_epochs=MAX_EPOCHS, early_stopping=3, 
                dl_train=dl_train, dl_test=dl_valid, device=device, dataset_name=dataset_name)
        elif model_name == 'VAE':
            pass

        return val_loss


    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler())
    study.optimize(objective, n_trials=n_trials)
    df_tuning_results = study.trials_dataframe()

    return study, df_tuning_results