import os

import torch
import wandb
from torch._C import dtype

from bcnf.utils import get_dir


def get_training_device() -> torch.device:
    """
    Returns the device for training. If a GPU is available, use it. Otherwise, use the CPU.

    Parameters
    ----------
    None

    Returns
    -------
    device : torch.device
        The device to use for training
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # TODO: FIX torch.set_default_dtype(torch.cuda.FloatTensor)
    else:
        device = torch.device('cpu')
        # TODO: FIX torch.set_default_dtype(torch.float)

    print(f"Using device: {device}")

    return device


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
    # Set data type
    print("Setting data type to: ", dtype)
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
