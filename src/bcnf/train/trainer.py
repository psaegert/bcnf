import datetime
import time
from typing import Callable

import torch
from dynaconf import Dynaconf
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from tqdm import tqdm

import wandb
# import wandb
from bcnf.train import TrainerDataHandler, TrainerLossHandler, TrainerModelHandler, TrainerScheduler, TrainerUtilities


class Trainer():
    def __init__(self,
                 config: Dynaconf,
                 project_name: str = None):

        self.config = config

        if project_name is None:
            raise ValueError("project_name is not set. Please use the project_name parameter to set the project name.")
        else:
            self.project_name = project_name

        # Initialize the data handler, model handler, and utilities
        print("Initializing Trainer...")
        self.utilities = TrainerUtilities()
        self.data_handler = TrainerDataHandler()
        self.model_handler = TrainerModelHandler()
        self.scheduler_creator = TrainerScheduler()
        self.loss_handler = TrainerLossHandler(val_loss_alpha=self.config["training"]["val_loss_alpha"],
                                               val_loss_patience=self.config["training"]["val_loss_patience"],
                                               val_loss_tolerance_mode=self.config["training"]["val_loss_tolerance_mode"],
                                               val_loss_tolerance=self.config["training"]["val_loss_tolerance"])

        self.tensor_size = self.utilities.set_data_types(tensor_size=self.config["model"]["tensor_size"])
        self.device = self.utilities.get_training_device()

        print("Initialisation complete")
        print("-----------------------------------------------------------------------------")

    def training_pipeline(self) -> None:
        # tell wandb to get started
        with wandb.init(project=self.project_name,  # type: ignore
                        config=self.config.as_dict()):  # type: ignore

            # access all HPs through wandb.config, so logging matches execution!
            self.config = wandb.config  # type: ignore
            # Convert wandb config keys to lowercase
            self.config = {k.lower(): v for k, v in wandb.config.as_dict().items()}  # type: ignore

            # make the model, data, and optimization problem
            model, dataset, loss_function, optimizer, scheduler = self._make()
            print("\nCreated all nessesary objects\n")

            if (self.config["training"]["cross_validation"]):
                model = self._train_kfold(model, dataset, loss_function, optimizer, scheduler)
            else:
                model = self._normal_training(model, dataset, loss_function, optimizer, scheduler)

        return model

    def _make(self) -> tuple[torch.nn.Module,
                             torch.utils.data.TensorDataset,
                             torch.nn.Module,
                             torch.optim.Optimizer,
                             torch.optim.lr_scheduler.ReduceLROnPlateau]:
        # Make the data
        data = self.data_handler.get_data_for_training(config=self.config["data"],
                                                       data_type=self.tensor_size)

        # Make the model
        model = self.model_handler.make_model(config=self.config["model"],
                                              data_size_primary=data[0][1].shape,
                                              data_size_feature=data[0][0].shape,
                                              device=self.device,
                                              data_type=self.tensor_size)

        # Verify the objects
        if self.config["training"]["verify"]:
            print("Please verify the model and data")
            self.model_handler.verify_model(model, data[0])
            self.data_handler.verify_data(data)
            input("Press Enter to continue...")
        else:
            print("Verification skipped")

        # loss and optimizer
        loss_function = self.model_handler.inn_nll_loss
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.config["training"]["learning_rate"])

        scheduler = self.scheduler_creator._create_scheduler(optimizer)

        return model, data, loss_function, optimizer, scheduler

    def _train_kfold(self,
                     model: torch.nn.Module,
                     dataset: torch.utils.data.TensorDataset,
                     loss_function: Callable,
                     optimizer: torch.optim.Optimizer,
                     scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau) -> torch.nn.Module:

        # Train the model with k-fold cross-validation
        kf = KFold(n_splits=self.config["training"]["n_folds"],
                   shuffle=self.config["training"]["shuffle"],
                   random_state=self.config["training"]["random_state"])
        self.fold_metrics: list = []

        indices = list(range(len(dataset)))
        for train_index, val_index in kf.split(indices):
            train_subset = Subset(dataset, train_index)
            val_subset = Subset(dataset, val_index)

            # create the dataloaders
            train_loader = self.data_handler.make_data_loader(dataset=train_subset,
                                                              batch_size=self.config["training"]["batch_size"],
                                                              pin_memory=self.config["training"]["pin_memory"],
                                                              num_workers=self.config["training"]["num_workers"])
            test_loader = self.data_handler.make_data_loader(dataset=val_subset,
                                                             batch_size=self.config["training"]["batch_size"],
                                                             pin_memory=self.config["training"]["pin_memory"],
                                                             num_workers=self.config["training"]["num_workers"])

            # and use them to train the model
            model = self._train(model,
                                train_loader,
                                test_loader,
                                loss_function,
                                optimizer,
                                scheduler)

        return model

    def _normal_training(self,
                         model: torch.nn.Module,
                         dataset: torch.utils.data.TensorDataset,
                         loss_function: Callable,
                         optimizer: torch.optim.Optimizer,
                         scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau) -> torch.nn.Module:
        """
        Train the model on the given dataset once

        Parameters
        ----------
        model : torch.nn.Module
            The model to train
        dataset : torch.utils.data.TensorDataset
            The dataset to train on
        loss_function : Callable
            The loss function to use
        optimizer : torch.optim.Optimizer
            The optimizer to use
        scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
            The scheduler to use

        Returns
        -------
        model : torch.nn.Module
            The trained model
        """
        # Split the dataset
        train_subset, val_subset = self.data_handler.split_dataset(dataset,
                                                                   self.config["training"]["validation_split"])

        # create the dataloaders
        train_loader = self.data_handler.make_data_loader(dataset=train_subset,
                                                          batch_size=self.config["training"]["batch_size"],
                                                          pin_memory=self.config["training"]["pin_memory"],
                                                          num_workers=self.config["training"]["num_workers"])
        test_loader = self.data_handler.make_data_loader(dataset=val_subset,
                                                         batch_size=self.config["training"]["batch_size"],
                                                         pin_memory=self.config["training"]["pin_memory"],
                                                         num_workers=self.config["training"]["num_workers"])

        # and use them to train the model
        model = self._train(model,
                            train_loader,
                            test_loader,
                            loss_function,
                            optimizer,
                            scheduler)

        return model

    def _train(self,
               model: torch.nn.Module,
               train_loader: torch.utils.data.DataLoader,
               val_loader: torch.utils.data.DataLoader,
               loss_function: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau) -> torch.nn.Module:
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
        model : torch.nn.Module
            The trained model
        """

        '''
        wandb.watch(model,
                    loss_function,
                    log="all",
                    log_freq=self.config["training"]["wandb"]["model_log_frequency"])
        '''
        pbar = tqdm(range(self.config["training"]["n_epochs"]), disable=not self.config["training"]["verbose"])
        start_time = time.time()

        # Train the model
        for epoch in pbar:
            # ---------------- TRAINING MODE -------------------
            model.train()
            train_loss = 0.0

            for i, (x, y) in enumerate(train_loader):
                # Add the loss to the total
                train_loss += self._train_batch(x,
                                                y,
                                                model,
                                                optimizer,
                                                loss_function)

            # Calculate the average train loss
            train_loss /= len(train_loader)
            # Add the loss to the history
            self.loss_handler.loss_history["train"].append((epoch + 1, train_loss))
            wandb.log({"epoch": epoch, "train_loss": train_loss}, step=epoch)  # type: ignore

            # ---------------- VALIDATION MODE -------------------
            model.eval()
            with torch.no_grad():
                val_loss = 0.0

                for x, y in val_loader:
                    # Add the loss to the total
                    val_loss += self._validate_epoch(x,
                                                     y,
                                                     model,
                                                     loss_function)

                # Calculate the average loss
                val_loss /= len(val_loader)

            # Add the loss to the history
            self.loss_handler.loss_history["val"].append((epoch + 1, val_loss))
            wandb.log({"epoch": epoch, "val_loss": val_loss}, step=epoch)  # type: ignore

            # Update the rolling average of the validation loss
            if self.loss_handler.val_loss_rolling_avg is None:
                self.loss_handler.val_loss_rolling_avg = val_loss
            else:
                self.loss_handler.val_loss_rolling_avg = self.loss_handler.val_loss_alpha * \
                    self.loss_handler.val_loss_rolling_avg + \
                    (1 - self.loss_handler.val_loss_alpha) * val_loss

            # Add the learning rate and early stop counter to the history
            self.loss_handler.loss_history["lr"].append((epoch + 1, optimizer.param_groups[0]['lr']))
            self.loss_handler.loss_history["early_stop_counter"].append((epoch + 1, epoch - self.loss_handler.best_val_epoch))
            self.loss_handler.loss_history["time"].append((epoch + 1, datetime.datetime.now().timestamp()))

            pbar.set_description(f"Train: {train_loss:.4f} - Val: {val_loss:.4f} (avg: {self.loss_handler.val_loss_rolling_avg:.4f}, min: {self.loss_handler.best_val_loss:.4f}) | lr: {optimizer.param_groups[0]['lr']:.2e} - Patience: {epoch - self.loss_handler.best_val_epoch}/{self.loss_handler.val_loss_patience}")

            # Step the scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(self.loss_handler.val_loss_rolling_avg)
                else:
                    scheduler.step()

            # Check if the validation loss did not decrease in the last `val_loss_patience` epochs
            if self.loss_handler.val_loss_patience is not None and self.loss_handler.val_loss_rolling_avg is not None:
                if self.loss_handler.val_loss_tolerance_mode == "rel":
                    if self.loss_handler.val_loss_rolling_avg < self.loss_handler.best_val_loss * (1 - self.loss_handler.val_loss_tolerance):
                        best_val_loss = self.loss_handler.val_loss_rolling_avg
                        best_val_epoch = epoch
                elif self.loss_handler.val_loss_tolerance_mode == "abs":
                    if self.loss_handler.val_loss_rolling_avg < best_val_loss - self.loss_handler.val_loss_tolerance:
                        best_val_loss = self.loss_handler.val_loss_rolling_avg
                        best_val_epoch = epoch

                if (epoch - best_val_epoch) >= self.loss_handler.val_loss_patience:
                    self.loss_handlerloss_history["stop_reason"] = "val_loss_plateau"  # type: ignore
                    return self.loss_handler.loss_history

            # Check if the timeout has been reached
            if self.config["training"]["timeout"] is not None and time.time() - start_time > self.config["training"]["timeout"]:
                self.loss_handlerloss_history["stop_reason"] = "timeout"  # type: ignore
                return self.loss_handler.loss_history

        self.loss_handler.loss_history["stop_reason"] = "max_epochs"  # type: ignore

        return model

    def _train_batch(self,
                     x: torch.Tensor,
                     y: torch.Tensor,
                     model: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     loss_function: Callable) -> float:
        x, y = x.to(self.device), y.to(self.device)

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

    def _validate_epoch(self,
                        x: torch.Tensor,
                        y: torch.Tensor,
                        model: torch.nn.Module,
                        loss_function: Callable) -> float:
        # Move the data to the correct device
        x = x.to(model.device)
        y = y.to(model.device)

        # Run the model
        z = model.forward(y, x, log_det_J=True)

        # Calculate the loss
        loss = loss_function(z, model.log_det_J)

        return loss.item()
