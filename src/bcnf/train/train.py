import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from bcnf.model import CondRealNVP


def inn_nll_loss(z: torch.Tensor, log_det_J: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    if reduction == 'mean':
        return torch.mean(0.5 * torch.sum(z**2, dim=1) - log_det_J)
    else:
        return 0.5 * torch.sum(z**2, dim=1) - log_det_J


def train_CondRealNVP(
        model: CondRealNVP,
        optimizer: torch.optim.Optimizer,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        n_epochs: int = 1,
        X_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
        batch_size: int = 16,
        loss_history: dict[str, list] | None = None,
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
        The number of epochs to train for
    X_val : torch.Tensor | None
        The validation data (parameters)
    y_val : torch.Tensor | None
        The validation conditions (simulation)
    batch_size : int
        The batch size to use
    loss_history : dict | None
        The loss history to append to
    verbose : bool
        Whether to display a progress bar

    Returns
    -------
    dict
        The final loss history
    """
    # Create the dataloader
    datasetTrain = TensorDataset(X_train, y_train)
    train_loader = DataLoader(datasetTrain, batch_size=batch_size, shuffle=True)

    # Create the validation dataloader
    if X_val is not None and y_val is not None:
        datasetVal = TensorDataset(X_val, y_val)
        val_loader = DataLoader(datasetVal, batch_size=batch_size, shuffle=False)

    loss_history = {
        "train": [],
        "val": []
    }

    pbar = tqdm(range(n_epochs), disable=not verbose)

    # Train the model
    for _ in pbar:
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            # Reset the gradients
            optimizer.zero_grad()

            # Run the model
            z = model.forward(y, x, log_det_J=True)

            # Calculate the loss
            forward_loss = inn_nll_loss(z, model.log_det_J)

            loss = forward_loss

            # Backpropagate
            loss.backward()

            # Update the weights
            optimizer.step()

            # Add the loss to the total
            train_loss += loss.item()

        # Calculate the average loss
        train_loss /= len(train_loader)

        # Add the loss to the history
        loss_history["train"].append(train_loss)

        # Calculate the loss on the validation set
        if X_val is not None:
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
                    forward_loss = inn_nll_loss(z, model.log_det_J)

                    loss = forward_loss

                    # Add the loss to the total
                    val_loss += loss.item()

                # Calculate the average loss
                val_loss /= len(val_loader)

                # Add the loss to the history
                loss_history["val"].append(val_loss)

            pbar.set_description(f"Train: {train_loss:.4f} - Val: {val_loss:.4f}")
        else:
            pbar.set_description(f"Train: {train_loss:.4f}")

    return loss_history
