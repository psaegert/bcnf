from collections import deque
from typing import Any, Deque

import wandb


class TrainerParameterHistoryHandler():
    def __init__(self,
                 val_loss_window_size: int,
                 val_loss_patience: int | None = None,
                 val_loss_tolerance_mode: str = "rel",
                 val_loss_tolerance: float = 1e-3,
                 fold: int = 1) -> None:
        if val_loss_tolerance_mode not in ["rel", "abs"]:
            raise ValueError("val_loss_tolerance_mode must be either 'rel' or 'abs'")
        else:
            self.val_loss_tolerance_mode = val_loss_tolerance_mode

        self.best_val_loss = float('inf')
        self.best_val_epoch = 0

        # For scheduler
        self.val_losses: Deque[float] = deque(maxlen=val_loss_window_size)
        self.val_loss_rolling_avg: float
        self.val_loss_window_size = val_loss_window_size
        self.val_loss_patience = val_loss_patience
        self.val_loss_tolerance = val_loss_tolerance

        self.parameter_history: dict = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "distance_to_last_best_val_loss": [],
            "time": []
        }
        self.epoch = 0
        self.fold = fold

    def update_epoch(self, epoch: int) -> None:
        """
        Updates the epoch counter

        Parameters
        ----------
        epoch : int
            The epoch to be set

        Returns
        -------
        None
        """
        self.epoch = epoch

    def update_parameter_history(self,
                                 parameter: str,
                                 value: Any) -> None:

        self.parameter_history[parameter].append((self.epoch + 1, value))
        wandb.log({"epoch": self.epoch + 1, f"{parameter}_fold_{self.fold}": value}, step=self.epoch)  # type: ignore

    def update_rolling_validation_loss(self, val_loss: float) -> None:
        """
        Updates the rolling average of the validation loss

        Parameters
        ----------
        val_loss : float
            The validation loss to be added to the rolling average

        Returns
        -------
        None
        """
        if len(self.val_losses) < self.val_loss_window_size:
            self.val_losses.append(val_loss)
        else:
            self.val_losses.popleft()
            self.val_losses.append(val_loss)

        self.val_loss_rolling_avg = sum(self.val_losses) / len(self.val_losses)

    def update_scheduler_parameters(self) -> None:
        """
        Updates the scheduler parameters

        Returns
        -------
        None
        """
        # Check if the validation loss did not decrease in the last `val_loss_patience` epochs
        if self.val_loss_patience is not None and self.val_loss_rolling_avg is not None:
            if self.val_loss_tolerance_mode == "rel":
                if self.val_loss_rolling_avg < self.best_val_loss * (1 - self.val_loss_tolerance):
                    self.best_val_loss = self.val_loss_rolling_avg
                    self.best_val_epoch = self.epoch
            elif self.val_loss_tolerance_mode == "abs":
                if self.val_loss_rolling_avg < self.best_val_loss - self.val_loss_tolerance:
                    self.best_val_loss = self.val_loss_rolling_avg
                    self.best_val_epoch = self.epoch
