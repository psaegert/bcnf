import wandb
import torch

from bcnf.utils import get_dir
import os


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
