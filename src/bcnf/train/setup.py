import wandb
import torch
import os
from torch.utils.data import TensorDataset

from bcnf.utils import get_dir
from bcnf.simulation.physics import get_data
from bcnf.models import FullyConnectedFeatureNetwork, CondRealNVP


def set_training_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type(torch.FloatTensor)

    return device


def wandb_login(filename: str = "wandbAPIKey.txt") -> None:
    key_file = get_dir(filename)
    
    if not os.path.exists(key_file):
        raise FileNotFoundError(f"File '{key_file}' does not exist.")
        
    with open(key_file, 'r') as f:
        API_key = f.read().strip()
        
    wandb.login(key=API_key)


def get_data_for_training(config: dict) -> torch.utils.data.TensorDataset:
    
    X, y = get_data(
    T=config.T,
    dt=config.dt,
    N=config.N,
    break_on_impact=config.break_on_impact,
    )

    X_tensor = torch.Tensor(X.reshape(X.shape[0], -1))
    y_tensor = torch.Tensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    
    return dataset


def make_loader(dataset: torch.tensor, batch_size: int, num_workers: int = 2) -> torch.utils.data.DataLoader:
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True,
                                         num_workers=num_workers)
    return loader


def make_model(config: dict, device: torch.device) -> torch.nn.Module:
    feature_network = FullyConnectedFeatureNetwork(sizes=config.sizes,
                                                   dropout=config.dropout).to(device)

    # Create the model
    model = CondRealNVP(size=config.size,
                        nested_sizes=config.nested_sizes,
                        n_blocks=config.n_blocks,
                        n_conditions=config.n_conditions,
                        feature_network=feature_network,
                        dropout=config.dropout,
                        act_norm=config.act_norm,
                        device=device).to(device)
    
    return model


def inn_nll_loss(z: torch.Tensor, log_det_J: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    if reduction == 'mean':
        return torch.mean(0.5 * torch.sum(z**2, dim=1) - log_det_J)
    else:
        return 0.5 * torch.sum(z**2, dim=1) - log_det_J