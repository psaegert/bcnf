import datetime
import time
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from bcnf.errors import TrainingDivergedError
from bcnf.factories import OptimizerFactory, SchedulerFactory
from bcnf.models.cnf import CondRealNVP_v2
from bcnf.train.trainer_data_handler import TrainerDataHandler
from bcnf.train.trainer_loss_handler import TrainerParameterHistoryHandler
from bcnf.train.utils import get_data_type
from bcnf.utils import ParameterIndexMapping, inn_nll_loss


class Trainer():
    def __init__(self, config: dict, project_name: str, parameter_index_mapping: ParameterIndexMapping, hybrid_weight: float = 0, verbose: bool = False) -> None:
        self.config = config
        self.verbose = verbose
        self.project_name = project_name
        self.parameter_index_mapping = parameter_index_mapping
        self.hybrid_weight = hybrid_weight

        # Initialize the data handler, model handler, and utilities
        self.data_handler = TrainerDataHandler()
        self.meta_scheduler = TrainerParameterHistoryHandler(
            val_loss_window_size=self.config["training"]["val_loss_window_size"],
            val_loss_patience=self.config["training"]["val_loss_patience"],
            val_loss_tolerance_mode=self.config["training"]["val_loss_tolerance_mode"],
            val_loss_tolerance=self.config["training"]["val_loss_tolerance"])

        self.dtype = get_data_type(dtype=self.config["global"]["dtype"])

        if self.verbose:
            print(f'Using dtype: {self.dtype}')

        self.data = self.data_handler.get_data_for_training(
            config=self.config,
            dtype=self.dtype,
            parameter_index_mapping=self.parameter_index_mapping,
            verbose=self.verbose)

        self.loss_function = inn_nll_loss

    def train(self, model: CondRealNVP_v2) -> CondRealNVP_v2:
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
            parameters=model.parameters(),
            optimizer_kwargs=self.config['optimizer']['kwargs'])

        scheduler = SchedulerFactory.get_scheduler(
            scheduler=self.config['lr_scheduler']['type'],
            optimizer=optimizer,
            scheduler_kwargs=self.config['lr_scheduler']['kwargs'])

        with wandb.init(project=self.project_name, config=self.config, entity="bcnf"):  # type: ignore

            # access all HPs through wandb.config, so logging matches execution!
            self.config = wandb.config  # type: ignore

            # Convert wandb config keys to lowercase
            self.config = {k.lower(): v for k, v in wandb.config.items()}  # type: ignore

            # Split the dataset
            train_subset, val_subset = self.data_handler.split_dataset(
                self.data,
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

            self.mse_loss = torch.nn.MSELoss()

            # and use them to train the model
            model = self._train(
                model,
                train_loader,
                val_loader,
                self.loss_function,
                optimizer,
                scheduler)

            return model

    def _train(
            self,
            model: CondRealNVP_v2,
            train_loader: DataLoader,
            val_loader: DataLoader,
            loss_function: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
            **kwargs: Any) -> CondRealNVP_v2:
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

        Returns
        -------
        ConditionalInvertibleLayer
            The trained model
        """
        self.meta_scheduler = TrainerParameterHistoryHandler(
            val_loss_window_size=self.config["training"]["val_loss_window_size"],
            val_loss_patience=self.config["training"]["val_loss_patience"],
            val_loss_tolerance_mode=self.config["training"]["val_loss_tolerance_mode"],
            val_loss_tolerance=self.config["training"]["val_loss_tolerance"],
            fold=kwargs.get("fold", -1))

        pbar = tqdm(range(self.config["training"]["n_epochs"]), disable=not self.config["training"]["verbose"])
        start_time = time.time()

        # Train the model
        for epoch in pbar:
            self.meta_scheduler.update_epoch(epoch)
            # ---------------- TRAINING MODE -------------------
            model.train()
            train_loss = 0.0
            train_loss_mse = 0.0
            train_loss_nll = 0.0

            for i, data in enumerate(train_loader):
                y, *conditions = data
                loss, nll_loss, mse_loss = self._train_batch(y, *conditions, model=model, optimizer=optimizer, loss_function=loss_function)

                if loss > 1e5 or np.isnan(loss):
                    raise TrainingDivergedError(f"Loss exploded to {loss} at epoch {epoch + i / len(train_loader)}")

                train_loss += loss
                train_loss_mse += mse_loss
                train_loss_nll += nll_loss

            # Calculate the average train loss
            train_loss /= len(train_loader)
            train_loss_mse /= len(train_loader)
            train_loss_nll /= len(train_loader)

            # ---------------- VALIDATION MODE -------------------
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_loss_mse = 0.0
                val_loss_nll = 0.0

                for data in val_loader:
                    y, *conditions = data
                    loss, nll_loss, mse_loss, z_mean, z_std = self._validate_batch(y, *conditions, model=model, loss_function=loss_function)

                    val_loss += loss
                    val_loss_mse += mse_loss
                    val_loss_nll += nll_loss

                # Calculate the average loss
                val_loss /= len(val_loader)
                val_loss_mse /= len(val_loader)
                val_loss_nll /= len(val_loader)

            self.meta_scheduler.update_rolling_validation_loss(val_loss)

            # Update the parameter history
            self.meta_scheduler.log("train_loss", train_loss)
            self.meta_scheduler.log("train_loss_mse", train_loss_mse)
            self.meta_scheduler.log("train_loss_nll", train_loss_nll)
            self.meta_scheduler.log("val_loss", val_loss)
            self.meta_scheduler.log("val_loss_mse", val_loss_mse)
            self.meta_scheduler.log("val_loss_nll", val_loss_nll)

            self.meta_scheduler.log("lr", optimizer.param_groups[0]['lr'])
            self.meta_scheduler.log("distance_to_last_best_val_loss", epoch - self.meta_scheduler.best_val_epoch)
            self.meta_scheduler.log("time", datetime.datetime.now().timestamp())
            self.meta_scheduler.log("z_mean_mean", z_mean.mean().item())
            self.meta_scheduler.log("z_mean_std", z_mean.std().item())
            self.meta_scheduler.log("z_std_mean", z_std.mean().item())
            self.meta_scheduler.log("z_std_std", z_std.std().item())
            self.meta_scheduler.log("log_det_J", model.log_det_J.mean().item())

            # Update the description of the progress bar
            pbar.set_description(f"Train: {train_loss:.3f} - Val: {val_loss:.3f} (avg: {self.meta_scheduler.val_loss_rolling_avg:.3f}, min: {self.meta_scheduler.best_val_loss:.3f}) | lr: {optimizer.param_groups[0]['lr']:.2e} - Patience: {epoch - self.meta_scheduler.best_val_epoch}/{self.meta_scheduler.val_loss_patience} - z: ({z_mean.mean().item():.3f} ± {z_mean.std().item():.3f}) ± ({z_std.mean().item():.3f} ± {z_std.std().item():.3f}) - ldJ: {model.log_det_J.mean().item():.2f}")

            # Step the scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(self.meta_scheduler.val_loss_rolling_avg)
                elif isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
                    scheduler.step()

            self.meta_scheduler.update_best_loss()

            if self.meta_scheduler.parameter_history["distance_to_last_best_val_loss"][-1][1] >= self.meta_scheduler.val_loss_patience:
                self.meta_scheduler.parameter_history["stop_reason"] = "val_loss_plateau"
                return model

            # Check if the timeout has been reached
            if self.config["training"]["timeout"] is not None and time.time() - start_time > self.config["training"]["timeout"]:
                self.meta_scheduler.parameter_history["stop_reason"] = "timeout"
                return model

        self.meta_scheduler.parameter_history["stop_reason"] = "max_epochs"

        return model

    def _train_batch(
            self,
            y: torch.Tensor,
            *conditions: torch.Tensor,
            model: CondRealNVP_v2,
            optimizer: torch.optim.Optimizer,
            loss_function: Callable) -> tuple[float, float, float]:

        optimizer.zero_grad()

        # Run the model
        z, h = model.forward(
            y.to(model.device),
            *[c.to(model.device) for c in conditions],
            log_det_J=True,
            return_features=True)

        if self.hybrid_weight > 0:
            y_hat = model.prediction_head(h)
            mse_loss = self.mse_loss(y_hat, y)
        else:
            mse_loss = torch.tensor(0.0)

        nll_loss = loss_function(z, model.log_det_J)

        loss = (nll_loss + mse_loss * self.hybrid_weight) / (1 + self.hybrid_weight)

        loss.backward()

        optimizer.step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        return loss.item(), nll_loss.item(), mse_loss.item()

    def _validate_batch(
            self,
            y: torch.Tensor,
            *conditions: torch.Tensor,
            model: CondRealNVP_v2,
            loss_function: Callable) -> tuple[float, float, float, torch.Tensor, torch.Tensor]:

        # Run the model
        z, h = model.forward(
            y.to(model.device),
            *[c.to(model.device) for c in conditions],
            log_det_J=True,
            return_features=True)

        if self.hybrid_weight > 0:
            y_hat = model.prediction_head(h)
            mse_loss = self.mse_loss(y_hat, y)
        else:
            mse_loss = torch.tensor(0.0)

        nll_loss = loss_function(z, model.log_det_J)

        loss = (nll_loss + mse_loss * self.hybrid_weight) / (1 + self.hybrid_weight)

        return loss.item(), nll_loss.item(), mse_loss.item(), z.mean(dim=0), z.std(dim=0)
