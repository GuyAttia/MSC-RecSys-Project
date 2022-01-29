import torch
from os import path
from ignite.engine import Engine, Events
from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

from src.loss import *


def train_mf(model, optimizer, max_epochs, early_stopping, dl_train, dl_test, device, dataset_name):
    model = model.to(device)
    criterion = RMSELoss()

    def train_step(engine, batch):
        users, items, y = batch
        users.to(device)
        items.to(device)
        y = y.float()
        y_pred = model(users, items)
        loss = criterion(y_pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    trainer = Engine(train_step)

    def validation_step(engine, batch):
        with torch.no_grad():
            users, items, y = batch
            users.to(device)
            items.to(device)
            y_pred = model(users, items)
            return y_pred, y

    # Generate training and validation evaluators to print results during running
    val_metrics = {
        "loss": Loss(criterion)
    }
    train_evaluator = Engine(validation_step)
    val_evaluator = Engine(validation_step)
    # Attach metrics to the evaluators
    val_metrics['loss'].attach(train_evaluator, 'loss')
    val_metrics['loss'].attach(val_evaluator, 'loss')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(dl_train)
        metrics = train_evaluator.state.metrics
        print(f"Training Results - Epoch[{trainer.state.epoch}] Avg loss: {metrics['loss']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(dl_test)
        metrics = val_evaluator.state.metrics
        print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg loss: {metrics['loss']:.2f}")

    # Model Checkpoint
    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss

    checkpoint_dir = path.join("checkpoints", dataset_name)

    # Checkpoint to store n_saved best models wrt score function
    model_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        n_saved=1,
        filename_prefix="best_mf",
        score_function=score_function,
        score_name='neg_loss',
        global_step_transform=global_step_from_engine(trainer), # helps fetch the trainer's state
        require_empty=False
    )
    val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})
    
    # Early stopping
    if early_stopping > 0:
        handler = EarlyStopping(patience=early_stopping, score_function=score_function, trainer=trainer)
        val_evaluator.add_event_handler(Events.COMPLETED, handler)

    # Tensorboard logger
    tb_logger = TensorboardLogger(log_dir=path.join('tb-logger', dataset_name, 'MF'))
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=100),
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )
    for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator)]:
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
    # Return the best val score
    best_val_score = -handler.best_score
    return best_val_score


def train_autorec(model, optimizer, max_epochs, early_stopping, dl_train, dl_test, device, dataset_name):
    model = model.to(device)
    criterion = NON_ZERO_RMSELoss()

    def train_step(engine, batch):
        x = y = batch
        x.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    trainer = Engine(train_step)

    def validation_step(engine, batch):
        with torch.no_grad():
            x = y = batch
            x.to(device)
            y_pred = model(x)
            return y_pred, y

    # Generate training and validation evaluators to print results during running
    val_metrics = {
        "loss": Loss(criterion)
    }
    train_evaluator = Engine(validation_step)
    val_evaluator = Engine(validation_step)
    # Attach metrics to the evaluators
    val_metrics['loss'].attach(train_evaluator, 'loss')
    val_metrics['loss'].attach(val_evaluator, 'loss')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(dl_train)
        metrics = train_evaluator.state.metrics
        print(f"Training Results - Epoch[{trainer.state.epoch}] Avg loss: {metrics['loss']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(dl_test)
        metrics = val_evaluator.state.metrics
        print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg loss: {metrics['loss']:.2f}")

    # Model Checkpoint
    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss

    checkpoint_dir = path.join("checkpoints", dataset_name)

    # Checkpoint to store n_saved best models wrt score function
    model_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        n_saved=1,
        filename_prefix="best_autorec",
        score_function=score_function,
        score_name='neg_loss',
        global_step_transform=global_step_from_engine(trainer), # helps fetch the trainer's state
        require_empty=False
    )
    val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})
    
    # Early stopping
    if early_stopping > 0:
        handler = EarlyStopping(patience=early_stopping, score_function=score_function, trainer=trainer)
        val_evaluator.add_event_handler(Events.COMPLETED, handler)

    # Tensorboard logger
    tb_logger = TensorboardLogger(log_dir=path.join('tb-logger', dataset_name, 'AutoRec'))
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=100),
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )
    for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator)]:
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
    # Return the best val score
    best_val_score = -handler.best_score
    return best_val_score
