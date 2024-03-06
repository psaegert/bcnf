from typing import Type

import torch
from sklearn.model_selection import KFold

from bcnf.models import ConditionalInvertibleLayer
from bcnf.models.feature_network import FeatureNetwork
from bcnf.train import train_CondRealNVP


def cross_validate(
        model_class: Type[ConditionalInvertibleLayer],
        model_kwargs: dict,
        feature_network_class: Type[FeatureNetwork],
        feature_network_kwargs: dict,
        optimizer_class: Type[torch.optim.Optimizer],
        optimizer_kwargs: dict,
        lr_scheduler_class: Type[torch.optim.lr_scheduler._LRScheduler],
        lr_scheduler_kwargs: dict,
        X: torch.Tensor,
        y: torch.Tensor,
        n_splits: int = 5,
        n_epochs: int = 1,
        val_loss_patience: float | None = None,
        val_loss_tolerance: float = 1e-3,
        batch_size: int = 64,
        device: str = 'cpu',
        verbose: bool = True,
        shuffle: bool = False,
        random_state: int | None = None) -> list[dict[str, dict[str, list] | float]]:

    # Move the data to the device
    X = X.to(device)
    y = y.to(device)

    # Split the data into folds
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    fold_metrics = []

    for i, (train_index, val_index) in enumerate(kf.split(X)):

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Create the feature network
        feature_network = feature_network_class(**feature_network_kwargs).to(device)

        # Create the model
        model = model_class(**model_kwargs, feature_network=feature_network).to(device)

        # Create the optimizer
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

        # Create the learning rate scheduler
        lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_kwargs)

        # Train the model
        loss_history = train_CondRealNVP(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            X_train=X_train,
            y_train=y_train,
            n_epochs=n_epochs,
            val_loss_patience=val_loss_patience,
            val_loss_tolerance=val_loss_tolerance,
            X_val=X_val,
            y_val=y_val,
            batch_size=batch_size,
            verbose=verbose)

        fold_metrics.append({
            'loss_history': loss_history,
            'train_loss': loss_history['train'][-1],
            'val_loss': loss_history['val'][-1]
        })

        # TODO: Calibration metrics

    return fold_metrics
