import os

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
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type(torch.FloatTensor)

        return device

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

        wandb.login(key=API_key)
