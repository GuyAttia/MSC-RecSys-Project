from os import path

import torch
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Loss

from src.loss import *

# We used the Ignite package for smarter building of our trainers.
# This package provide built-in loggers and handlers for different actions.


def train_vae(model, optimizer, max_epochs, early_stopping, dl_train, dl_test, device, dataset_name, mrr_threshold=4):
    """
    Build a trainer for the VAE model
    """
    model = model.to(device)
    criterion = VAE_Loss()
    criterion2 = NON_ZERO_RMSELoss()

    def train_step(engine, batch):
        """
        Define the train step.
        Each sample in the batch is a user/item vector, which is also the target (what we want to reconstruct)
        """
        x = y = batch
        x.to(device)

        model.train()
        recon_x, mu, logvar = model(x)
        loss = criterion(recon_x, y, mu, logvar)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    trainer = Engine(train_step)    # Instantiate the trainer object

    def validation_step(engine, batch):
        """
        Define the validation step (without updating the parameters).
        """
        model.eval()

        with torch.no_grad():
            x = y = batch
            x.to(device)
            recon_x, mu, logvar = model(x)
            kwargs = {'mu': mu, 'logvar': logvar}
            return recon_x, y, kwargs

    def validation_step_mrr(engine, batch):
        """
        Define the validation step (without updating the parameters).
        """
        model.eval()

        with torch.no_grad():
            x = y = batch
            x.to(device)
            recon_x, mu, logvar = model(x)
            kwargs = {'mu': mu, 'logvar': logvar}
            recon_x_mrr = torch.clone(recon_x.detach()).to('cpu')
            y_mrr = torch.clone(y.detach()).to('cpu')
            return recon_x_mrr, y_mrr, kwargs

    # Generate training and validation evaluators to print results during running
    val_metrics = {
        "loss": Loss(criterion),
        'rmse_loss': Loss(criterion2)
    }
    mrr_metric = MetricMRR_Vec(mrr_threshold=mrr_threshold)

    train_evaluator = Engine(validation_step)
    val_evaluator = Engine(validation_step)
    val_mrr_evaluator = Engine(validation_step_mrr)
    # Attach metrics to the evaluators
    val_metrics['loss'].attach(train_evaluator, 'loss')
    val_metrics['loss'].attach(val_evaluator, 'loss')
    val_metrics['rmse_loss'].attach(val_evaluator, 'rmse_loss')
    mrr_metric.attach(val_mrr_evaluator, 'mrr')

    # Attach logger to print the training loss after each epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(dl_train)
        metrics = train_evaluator.state.metrics
        print(
            f"Training Results - Epoch[{trainer.state.epoch}] Avg loss: {metrics['loss']:.2f}")

    # Attach logger to print the validation loss after each epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(dl_test)
        metrics = val_evaluator.state.metrics
        print(
            f"Validation Results - Epoch[{trainer.state.epoch}] Avg loss: {metrics['loss']:.2f}  |  Avg RMSE: {metrics['rmse_loss']:.2f}")

    # Attach logger to print the validation loss after each epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_mrr_validation_results(trainer):
        val_mrr_evaluator.run(dl_test)
        metrics = val_mrr_evaluator.state.metrics
        print(
            f"Validation Results - Epoch[{trainer.state.epoch}] MRR: {metrics['mrr']:.2f}")

    # Model Checkpoint
    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss

    checkpoint_dir = path.join("checkpoints", dataset_name)

    # Checkpoint to store n_saved best models wrt score function
    model_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        n_saved=1,
        filename_prefix="best_vae",
        score_function=score_function,
        score_name='neg_loss',
        global_step_transform=global_step_from_engine(
            trainer),  # helps fetch the trainer's state
        require_empty=False
    )
    # After each epoch if the validation results are better - save the model as a file
    val_evaluator.add_event_handler(
        Events.COMPLETED, model_checkpoint, {"model": model})

    # Early stopping
    if early_stopping > 0:
        handler = EarlyStopping(patience=early_stopping,
                                score_function=score_function, trainer=trainer)
        val_evaluator.add_event_handler(Events.COMPLETED, handler)

    # Tensorboard logger - log the training and evaluation losses as function of the iterations & epochs
    tb_logger = TensorboardLogger(
        log_dir=path.join('tb-logger', dataset_name, 'VAE'))
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=100),
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )
    for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator), ('mrr_validation', val_mrr_evaluator)]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=global_step_from_engine(trainer),
        )

    # Run the trainer
    trainer.run(dl_train, max_epochs=max_epochs)

    tb_logger.close()   # Close logger
    # Return the best validation score
    best_val_score = -handler.best_score
    return best_val_score


# Only for testing
if __name__ == '__main__':
    from src.data import *
    from src.hyperparams_tuning import *

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = 'books'
    mrr_threshold = 8

    max_epochs = 2
    model_name = 'VAE'
    best_params = {
        'learning_rate': 0.001,
        'p_dims': [200, 600],
        'activation_func': 'relu',
        'optimizer': "RMSprop",
        'dropout': 0.2,
        'batch_size': 512
    }
    dl_train, _, dl_test, _ = dataloaders(
        dataset_name=dataset_name, batch_size=best_params['batch_size'], device=device)
    model = get_model(model_name, best_params, dl_train)  # Build model
    optimizer = getattr(optim, best_params['optimizer'])(
        model.parameters(), lr=best_params['learning_rate'])  # Instantiate optimizer
    test_loss = train(model_name, model, optimizer, max_epochs, dl_train,
                      dl_test, device, dataset_name, mrr_threshold=mrr_threshold)
