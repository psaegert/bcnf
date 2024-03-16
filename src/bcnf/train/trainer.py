import datetime
import time
from typing import Any, Callable

import torch
import wandb
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

from bcnf.errors import TrainingDivergedError
from bcnf.factories import OptimizerFactory, SchedulerFactory
from bcnf.models.cnf import ConditionalInvertibleLayer
from bcnf.train.trainer_data_handler import TrainerDataHandler
from bcnf.train.trainer_loss_handler import TrainerParameterHistoryHandler
from bcnf.train.utils import get_data_type, get_training_device
from bcnf.utils import inn_nll_loss


class Trainer():
    def __init__(self, config: dict, project_name: str = None, verbose: bool = False) -> None:
        self.config = config

        if project_name is None:
            raise ValueError("project_name is not set. Please use the project_name parameter to set the project name.")
        else:
            self.project_name = project_name

        self.verbose = verbose

        # Initialize the data handler, model handler, and utilities
        if self.verbose:
            print("Initializing Trainer...")
        self.data_handler = TrainerDataHandler()
        self.history_handler = TrainerParameterHistoryHandler(
            val_loss_window_size=self.config["training"]["val_loss_window_size"],
            val_loss_patience=self.config["training"]["val_loss_patience"],
            val_loss_tolerance_mode=self.config["training"]["val_loss_tolerance_mode"],
            val_loss_tolerance=self.config["training"]["val_loss_tolerance"])

        self.dtype = get_data_type(dtype=self.config["model"]["dtype"])
        self.device = get_training_device()

        if self.verbose:
            print("Initialisation complete")
            print("-----------------------------------------------------------------------------")

    def make(self) -> tuple[TensorDataset, Callable]:
        # Make the data
        data = self.data_handler.get_data_for_training(
            config=self.config["data"],
            dtype=self.dtype)

        # loss and optimizer
        loss_function = inn_nll_loss

        return data, loss_function

    def train(self, model: ConditionalInvertibleLayer) -> ConditionalInvertibleLayer:
        """
        Train the model on the given dataset once

        Parameters
        ----------
        model : ConditionalInvertibleLayer
            The model to train

        Returns
        -------
        model : ConditionalInvertibleLayer
            The trained model
        """
        optimizer = OptimizerFactory.get_optimizer(
            optimizer=self.config['optimizer']['type'],
            model=model,
            optimizer_kwargs=self.config['optimizer']['kwargs'])

        scheduler = SchedulerFactory.get_scheduler(
            scheduler=self.config['lr_scheduler']['type'],
            optimizer=optimizer,
            scheduler_kwargs=self.config['lr_scheduler']['kwargs'])

        with wandb.init(project=self.project_name, config=self.config.as_dict()):  # type: ignore

            # access all HPs through wandb.config, so logging matches execution!
            self.config = wandb.config  # type: ignore

            # Convert wandb config keys to lowercase
            self.config = {k.lower(): v for k, v in wandb.config.as_dict().items()}  # type: ignore

            # make the model, data, and optimization problem
            data, loss_function = self.make()
            if self.verbose:
                print("\nCreated all nessesary objects\n")

            # Split the dataset
            train_subset, val_subset = self.data_handler.split_dataset(
                data,
                self.config["training"]["validation_split"])

            # create the dataloaders
            train_loader = self.data_handler.make_data_loader(
                dataset=train_subset,
                batch_size=self.config["training"]["batch_size"],
                pin_memory=self.config["training"]["pin_memory"],
                num_workers=self.config["training"]["num_workers"])

            val_loader = self.data_handler.make_data_loader(
                dataset=val_subset,
                batch_size=self.config["training"]["batch_size"],
                pin_memory=self.config["training"]["pin_memory"],
                num_workers=self.config["training"]["num_workers"])

            # and use them to train the model
            model = self._train(
                model,
                train_loader,
                val_loader,
                loss_function,
                optimizer,
                scheduler)

            return model

    def kfold_crossvalidation(self, model: torch.nn.Module) -> list[dict]:
        # Train the model with k-fold cross-validation
        kf = KFold(
            n_splits=self.config["training"]["n_folds"],
            random_state=self.config["training"]["random_state"])

        data, loss_function = self.make()
        if self.verbose:
            print("\nCreated all nessesary objects\n")

        fold_metrics: list = []
        indices = list(range(len(data)))
        for i, (train_index, val_index) in enumerate(kf.split(indices)):

            with wandb.init(project=self.project_name, config=self.config.as_dict()):  # type: ignore

                # access all HPs through wandb.config, so logging matches execution!
                self.config = wandb.config  # type: ignore

                # Convert wandb config keys to lowercase
                self.config = {k.lower(): v for k, v in wandb.config.as_dict().items()}  # type: ignore

                train_subset = Subset(data, train_index)
                val_subset = Subset(data, val_index)

                # create the dataloaders
                train_loader = self.data_handler.make_data_loader(
                    dataset=train_subset,
                    batch_size=self.config["training"]["batch_size"],
                    pin_memory=self.config["training"]["pin_memory"],
                    num_workers=self.config["training"]["num_workers"])

                val_loader = self.data_handler.make_data_loader(
                    dataset=val_subset,
                    batch_size=self.config["training"]["batch_size"],
                    pin_memory=self.config["training"]["pin_memory"],
                    num_workers=self.config["training"]["num_workers"])

                optimizer = OptimizerFactory.get_optimizer(
                    optimizer=self.config['optimizer']['type'],
                    model=model,
                    optimizer_kwargs=self.config['optimizer']['kwargs'])

                scheduler = SchedulerFactory.get_scheduler(
                    scheduler=self.config['lr_scheduler']['type'],
                    optimizer=optimizer,
                    scheduler_kwargs=self.config['lr_scheduler']['kwargs'])

                # and use them to train the model
                model, history = self._train(
                    model,
                    train_loader,
                    val_loader,
                    loss_function,
                    optimizer,
                    scheduler,
                    fold=i)

                fold_metrics.append(history)

        return fold_metrics

    def _train(
            self,
            model: torch.nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            loss_function: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
            **kwargs: Any) -> ConditionalInvertibleLayer:
        """
        Train the model on given dataloaders for training and validation

        Parameters
        ----------
        model : torch.nn.Module
            The model to train
        train_loader : torch.utils.data.DataLoader
            The dataloader for the training data
        val_loader : torch.utils.data.DataLoader
            The dataloader for the validation data
        loss_function : torch.nn.Module
            The loss function to use
        optimizer : torch.optim.Optimizer
            The optimizer to use
        scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
            The scheduler to use
        current_fold : int
            The current fold of the k-fold cross-validation

        Returns
        -------
        ConditionalInvertibleLayer
            The trained model
        """

        '''
        wandb.watch(model,
                    loss_function,
                    log="all",
                    log_freq=self.config["training"]["wandb"]["model_log_frequency"])
        '''
        self.history_handler = TrainerParameterHistoryHandler(
            val_loss_alpha=self.config["training"]["val_loss_window_size"],
            val_loss_patience=self.config["training"]["val_loss_patience"],
            val_loss_tolerance_mode=self.config["training"]["val_loss_tolerance_mode"],
            val_loss_tolerance=self.config["training"]["val_loss_tolerance"],
            fold=kwargs.get("fold", -1))

        pbar = tqdm(range(self.config["training"]["n_epochs"]), disable=not self.config["training"]["verbose"])
        start_time = time.time()

        # Train the model
        for epoch in pbar:
            self.history_handler.update_epoch(epoch)
            # ---------------- TRAINING MODE -------------------
            model.train()
            train_loss = 0.0

            for i, (x, y) in enumerate(train_loader):
                loss = self._train_batch(x, y, model, optimizer, loss_function)

                if loss > 1e5 or torch.isnan(loss).any():
                    raise TrainingDivergedError(f"Loss exploded to {loss} at epoch {epoch + i / len(train_loader)}")

                train_loss += loss

            # Calculate the average train loss
            train_loss /= len(train_loader)

            # ---------------- VALIDATION MODE -------------------
            model.eval()
            with torch.no_grad():
                val_loss = 0.0

                for x, y in val_loader:
                    loss, z_mean, z_std = self._validate_batch(
                        x,
                        y,
                        model,
                        loss_function)

                    val_loss += loss

                # Calculate the average loss
                val_loss /= len(val_loader)

            self.history_handler.update_rolling_validation_loss(val_loss)

            # Update the parameter history
            self.history_handler.update_parameter_history("train_loss", train_loss)
            self.history_handler.update_parameter_history("val_loss", val_loss)
            self.history_handler.update_parameter_history("lr", optimizer.param_groups[0]['lr'])
            self.history_handler.update_parameter_history("distance_to_last_best_val_loss", epoch - self.history_handler.best_val_epoch)
            self.history_handler.update_parameter_history("time", datetime.datetime.now().timestamp())

            # Update the description of the progress bar
            pbar.set_description(f"Train: {train_loss:.4f} - Val: {val_loss:.4f} (avg: {self.history_handler.val_loss_rolling_avg:.4f}, min: {self.history_handler.best_val_loss:.4f}) | lr: {optimizer.param_groups[0]['lr']:.2e} - Patience: {epoch - self.history_handler.best_val_epoch}/{self.history_handler.val_loss_patience} - z: {z_mean:.4f} ± {z_std:.4f}")

            # Step the scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(self.history_handler.val_loss_rolling_avg)
                elif isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
                    scheduler.step()

                self.history_handler.update_scheduler_parameters()

                if self.history_handler.parameter_history["distance_to_last_best_val_loss"][-1] >= self.history_handler.val_loss_patience:
                    self.history_handler.parameter_history["stop_reason"] = "val_loss_plateau"
                    return model, self.history_handler.parameter_history

            # Check if the timeout has been reached
            if self.config["training"]["timeout"] is not None and time.time() - start_time > self.config["training"]["timeout"]:
                self.history_handler.parameter_history["stop_reason"] = "timeout"
                return model, self.history_handler.parameter_history

        self.history_handler.parameter_history["stop_reason"] = "max_epochs"

        return model

    def _train_batch(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            model: ConditionalInvertibleLayer,
            optimizer: torch.optim.Optimizer,
            loss_function: Callable) -> float:

        x = x.to(model.device)
        y = y.to(model.device)

        # Forward pass ➡
        z = model.forward(y, x, log_det_J=True)
        loss = loss_function(z, model.log_det_J)

        # Backward pass ⬅
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Step with optimizer
        optimizer.step()

        return loss.item()

    def _validate_batch(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            model: ConditionalInvertibleLayer,
            loss_function: Callable) -> tuple[float, float, float]:
        # Move the data to the correct device
        x = x.to(model.device)
        y = y.to(model.device)

        # Run the model
        z = model.forward(y, x, log_det_J=True)

        # Calculate the loss
        loss = loss_function(z, model.log_det_J)

        return loss.item(), z.mean().item(), z.std().item()
