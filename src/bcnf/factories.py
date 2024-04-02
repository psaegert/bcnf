from typing import Any

import torch
import torch.nn as nn
import importlib

from bcnf.models.cnn import CNN
from bcnf.models.feature_network import ConcatenateCondition, FrExpFeatureNetwork, FeatureNetwork, FullyConnectedFeatureNetwork, LSTMFeatureNetwork, Transformer


class SchedulerFactory():
    @staticmethod
    def get_scheduler(scheduler: str, optimizer: torch.optim.Optimizer, scheduler_kwargs: Any) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        match scheduler:
            case "ReduceLROnPlateau":
                return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)
            case _:
                raise NotImplementedError(f"Scheduler {scheduler} not implemented")


class OptimizerFactory():
    @staticmethod
    def get_optimizer(optimizer: str, model: torch.nn.Module, optimizer_kwargs: Any) -> torch.optim.Optimizer:
        match optimizer:
            case "Adam":
                return torch.optim.Adam(model.parameters(), **optimizer_kwargs)
            case _:
                raise NotImplementedError(f"Optimizer {optimizer} not implemented")


class FeatureNetworkFactory():
    @staticmethod
    def get_feature_network(network: str | None, network_kwargs: Any) -> FeatureNetwork:
        match network:
            case "FullyConnected":
                return FullyConnectedFeatureNetwork(**network_kwargs)
            case "CNN":
                return CNN(**network_kwargs)
            case "LSTM":
                return LSTMFeatureNetwork(**network_kwargs)
            case "Transformer":
                return Transformer(**network_kwargs)
            case "ConcatenateCondition":
                return ConcatenateCondition(**network_kwargs)
            case "FrExpFeatureNetwork":
                return FrExpFeatureNetwork(**network_kwargs)
            case None:
                return nn.Identity()
            case _:
                raise NotImplementedError(f"Feature network {network} not implemented")


class LayerFactory():
    @staticmethod
    def get_layer(layer: str, *args: Any, **kwargs: Any) -> nn.Module:
        # Try to import the layer from torch.nn
        if hasattr(nn, layer):
            return getattr(nn, layer)(*args, **kwargs)

        # Try to import the layer from bcnf.models.layers
        bcnf_models = importlib.import_module("bcnf.models")
        if hasattr(bcnf_models, layer):
            return getattr(bcnf_models, layer)(*args, **kwargs)

        raise NotImplementedError(f"Layer {layer} not implemented")
