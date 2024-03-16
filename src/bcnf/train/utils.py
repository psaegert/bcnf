import os
from typing import Any

import torch
import wandb
from torch._C import dtype
from torchsummary import summary

from bcnf.utils import get_dir


def get_data_type(dtype: str) -> dtype:
    """
    Set the default data type to be used for training and models.

    Parameters
    ----------
    tensor_size : str
        The size of the tensor to use. Either "float32" or "float64"

    Returns
    -------
    torch_tensor_size : dtype
        The data type to use for the tensors
    """
    if dtype == "float32":
        torch_tensor_size = torch.float32
    elif dtype == "float64":
        torch_tensor_size = torch.float64
    else:
        print("tensor_size was not correctly specified in the config file, using default value 'float32'")
        torch_tensor_size = torch.float32

    return torch_tensor_size


def wandb_login(filename: str = "wandbAPIKey.txt") -> None:
    """
    Log into wandb using the API key stored in the file with the given filename.

    Parameters
    ----------
    filename : str
        The name of the file containing the API key for wandb. The default is "wandbAPIKey.txt"

    Returns
    -------
    None
    """
    key_file = get_dir(filename)

    if not os.path.exists(key_file):
        raise FileNotFoundError(f"File '{key_file}' does not exist.")

    with open(key_file, 'r') as f:
        API_key = f.read().strip()

    wandb.login(key=API_key)  # type: ignore


def model_summary(model: torch.nn.Module,
                  test_data: Any) -> None:
    """
    Print a summary of the model using the torchsummary package.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be summarized
    test_data : Any
        The test data to be used for the summary

    Returns
    -------
    None
    """
    summary(model, input_size=[test_data[1].shape, test_data[0].shape])
