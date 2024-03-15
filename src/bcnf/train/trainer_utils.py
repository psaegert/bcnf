import os
from typing import Any

import torch

import wandb
from bcnf.utils import get_dir


class TrainerUtilities():
    def __init__(self) -> None:
        pass

    def get_training_device(self) -> torch.device:
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

    def set_data_types(self,
                       tensor_size: str) -> Any:
        """
        Set the default data type to be used for training and models.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Set data type
        print("Setting data type to: ", tensor_size)
        if tensor_size == "float32":
            torch_tensor_size = torch.float32
        elif tensor_size == "float64":
            torch_tensor_size = torch.float64
        else:
            print("tensor_size was not correctly specified in the config file, using default value 'float32'")
            torch_tensor_size = torch.float32

        return torch_tensor_size

    def wandb_login(self, filename: str = "wandbAPIKey.txt") -> None:
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
