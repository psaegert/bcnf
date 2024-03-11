import datetime
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from bcnf.errors import TrainingDivergedError
from bcnf.models import CondRealNVP
from bcnf.train.trainer_utils import inn_nll_loss


def train_CondRealNVP(
        model: CondRealNVP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        n_epochs: int = 1,
        val_loss_patience: int | None = None,
        val_loss_tolerance: float = 1e-4,
        val_loss_tolerance_mode: str = "rel",
        val_loss_alpha: float = 0.95,
        timeout: float | None = None,
        X_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
        batch_size: int = 16,
        loss_history: dict[str, list[tuple[int | float, float]]] = None,
        verbose: bool = True) -> dict:
    """
    Train the model using the INN loss function

    Parameters
    ----------
    model : CondRealNVP
        The model to train
    optimizer : torch.optim.optimizer.Optimizer
        The optimizer to use
    X_train : torch.Tensor
        The training data (parameters)
    y_train : torch.Tensor
        The training conditions (simulation)
    n_epochs : int
        The maximum number of epochs to train for
    val_loss_patience : float
        The number of epochs to wait before stopping if the validation loss does not decrease
    val_loss_tolerance : float
        The minimum decrease in validation loss to consider as an improvement
    val_loss_tolerance_mode : str
        The mode to use for the tolerance (either 'rel' or 'abs')
    val_loss_alpha : float
        The exponential moving average factor for the validation loss
    timeout : float
        The maximum number of seconds to train for
    X_val : torch.Tensor | None
        The validation data (parameters)
    y_val : torch.Tensor | None
        The validation conditions (simulation)
    batch_size : int
        The batch size to use
    loss_history : dict[str, list[tuple[int | float, float]]] | None
        The loss history to append to. If None, a new dictionary will be created.
    verbose : bool
        Whether to display a progress bar

    Returns
    -------
    dict
        The final loss history
    """
    if val_loss_tolerance_mode not in ["rel", "abs"]:
        raise ValueError("val_loss_tolerance_mode must be either 'rel' or 'abs'")

    # Create the dataloader
    datasetTrain = TensorDataset(X_train, y_train)
    train_loader = DataLoader(datasetTrain, batch_size=batch_size, shuffle=True)

    # Create the validation dataloader
    if (X_val is not None and y_val is not None):
        do_validate = True
        datasetVal = TensorDataset(X_val, y_val)
        val_loader = DataLoader(datasetVal, batch_size=batch_size, shuffle=False)
        best_val_loss = float('inf')
        best_val_epoch = 0
        val_loss_rolling_avg = None
    else:
        do_validate = False

    if loss_history is None:
        loss_history = {
            "train": [],
            "val": [],
            "lr": [],
            "early_stop_counter": [],
            "time": []
        }
    elif isinstance(loss_history, dict):
        loss_history["train"] = []
        loss_history["val"] = []
        loss_history["lr"] = []
        loss_history["early_stop_counter"] = []
        loss_history["time"] = []

    pbar = tqdm(range(n_epochs), disable=not verbose)

    start_time = time.time()

    # Train the model
    for epoch in pbar:
        model.train()
        train_loss = 0.0

        for i, (x, y) in enumerate(train_loader):
            # Reset the gradients
            optimizer.zero_grad()

            # Run the model
            z = model.forward(y, x, log_det_J=True)

            # Calculate the loss
            loss = inn_nll_loss(z, model.log_det_J)

            # Backpropagate
            loss.backward()

            if loss.item() > 1e5 or torch.isnan(loss).any():
                raise TrainingDivergedError(f"Loss exploded to {loss.item()} at epoch {epoch + i / len(train_loader)}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update the weights
            optimizer.step()

            # Add the loss to the total
            train_loss += loss.item()

            # Add the loss to the history
            loss_history["train"].append((epoch + i / len(train_loader), loss.item()))

        # Calculate the average loss
        train_loss /= len(train_loader)

        # Calculate the loss on the validation set
        if do_validate:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0

                for x, y in val_loader:
                    # Move the data to the correct device
                    x = x.to(model.device)
                    y = y.to(model.device)

                    # Run the model
                    z = model.forward(y, x, log_det_J=True)

                    # Calculate the loss
                    loss = inn_nll_loss(z, model.log_det_J)

                    # Add the loss to the total
                    val_loss += loss.item()

                # Calculate the average loss
                val_loss /= len(val_loader)

                # Add the loss to the history
                loss_history["val"].append((epoch + 1, loss.item()))

                # Update the rolling average of the validation loss
                if val_loss_rolling_avg is None:
                    val_loss_rolling_avg = val_loss
                else:
                    val_loss_rolling_avg = val_loss_alpha * val_loss_rolling_avg + (1 - val_loss_alpha) * val_loss

            # Add the learning rate and early stop counter to the history
            loss_history["lr"].append((epoch + 1, optimizer.param_groups[0]['lr']))
            loss_history["early_stop_counter"].append((epoch + 1, epoch - best_val_epoch))
            loss_history["time"].append((epoch + 1, datetime.datetime.now().timestamp()))

            pbar.set_description(f"Train: {train_loss:.4f} - Val: {val_loss:.4f} (avg: {val_loss_rolling_avg:.4f}, min: {best_val_loss:.4f}) | lr: {optimizer.param_groups[0]['lr']:.2e} - Patience: {epoch - best_val_epoch}/{val_loss_patience}")
        else:
            pbar.set_description(f"Train: {train_loss:.4f}, lr: {optimizer.param_groups[0]['lr']:.2e}")

        # Step the scheduler
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(val_loss_rolling_avg)
            else:
                lr_scheduler.step()

        # Check if the validation loss did not decrease in the last `val_loss_patience` epochs
        if do_validate and val_loss_patience is not None and val_loss_rolling_avg is not None:
            if val_loss_tolerance_mode == "rel":
                if val_loss_rolling_avg < best_val_loss * (1 - val_loss_tolerance):
                    best_val_loss = val_loss_rolling_avg
                    best_val_epoch = epoch
            elif val_loss_tolerance_mode == "abs":
                if val_loss_rolling_avg < best_val_loss - val_loss_tolerance:
                    best_val_loss = val_loss_rolling_avg
                    best_val_epoch = epoch

            if (epoch - best_val_epoch) >= val_loss_patience:
                loss_history["stop_reason"] = "val_loss_plateau"  # type: ignore
                return loss_history

        # Check if the timeout has been reached
        if timeout is not None and time.time() - start_time > timeout:
            loss_history["stop_reason"] = "timeout"  # type: ignore
            return loss_history

    loss_history["stop_reason"] = "max_epochs"  # type: ignore

    return loss_history
