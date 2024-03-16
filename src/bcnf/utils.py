import os
import re
from typing import Iterator

import numpy as np
import torch
from dynaconf import Dynaconf


def load_config(config_file: str) -> dict:
    """
    Load a configuration file.

    Parameters
    ----------
    config_file : str
        The path to the configuration file.

    Returns
    -------
    config : dict
        The configuration dictionary.
    """
    if not isinstance(config_file, str):
        raise TypeError("config_file must be a string.")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"File '{config_file}' does not exist.")

    config = Dynaconf(settings_files=[os.path.join(get_dir("configs"), "trainer_config.yaml")])

    config.data['path'] = sub_root_path(config.data['path'])
    config.data['config_file'] = sub_root_path(config.data['config_file'])

    return config


def inn_nll_loss(z: torch.Tensor, log_det_J: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    if reduction == 'mean':
        return torch.mean(0.5 * torch.sum(z**2, dim=1) - log_det_J)
    else:
        return 0.5 * torch.sum(z**2, dim=1) - log_det_J


def get_gaussian_kernel(sigma: float, window_size: int = None) -> np.ndarray:
    if window_size is None:
        window_size = int(sigma * 3.5)

    return np.exp(-np.arange(-window_size, window_size + 1)**2 / (2 * sigma**2))


def partconv1d(data: np.ndarray, kernel: np.ndarray, periodic: bool = False) -> np.ndarray:
    '''
    Convolve data with a kernel, handling edges with appropriately normalized truncated kernel,
    optionally using periodic padding.

    Parameters
    ----------
    data : np.ndarray
        Data to be convolved.
    kernel : np.ndarray
        Kernel for convolution.
    periodic : bool, optional
        If True, the data is treated as circular and padded accordingly. Default is False.

    Returns
    -------
    np.ndarray
        Convolved data.
    '''
    if not isinstance(data, np.ndarray) or not isinstance(kernel, np.ndarray):
        raise ValueError("Data and kernel must be numpy arrays.")

    if len(kernel) % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    window_size = len(kernel) // 2

    if periodic:
        # Extend the data on both sides for circular data
        data = np.concatenate((data[-window_size:], data, data[:window_size]))

    # Convolve the middle section of the data with the kernel, i.e. where the data and the kernel overlap completely
    data_convolved_middle = np.convolve(data, kernel / kernel.sum(), mode='valid')

    # Convolve the edges of the data with the kernel, i.e. where the data and the kernel overlap partially
    data_convolved_left = np.empty(2 * window_size - 1)
    data_convolved_right = np.empty(2 * window_size - 1)
    for i in range(1, 2 * window_size):
        data_convolved_left[i - 1] = data[:i] @ kernel[-i:] / kernel[-i:].sum()
        data_convolved_right[i - 1] = data[- 2 * window_size + i:] @ kernel[:2 * window_size - i] / kernel[:2 * window_size - i].sum()

    # Convolve the data with the kernel
    data_convolved = np.concatenate((data_convolved_left[window_size - 1:], data_convolved_middle, data_convolved_right[:window_size]))

    if periodic:
        # Cut off the excess data
        data_convolved = data_convolved[window_size:-window_size]

    return data_convolved


def get_dir(*args: str, create: bool = False) -> str:
    """
    Get the path to the data directory.

    Parameters
    ----------
    args : str
        The path to the data directory.
    create : bool, optional
        Whether to create the directory if it does not exist, by default False.

    Returns
    -------
    str
        The path to the data directory.
    """
    if any([not isinstance(arg, str) for arg in args]):
        raise TypeError("All arguments must be strings.")

    if create:
        os.makedirs(os.path.join(os.path.dirname(__file__), '..', '..', *args), exist_ok=True)

    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', *args))


def sub_root_path(path: str) -> str:
    """
    Replace {{BCNF_ROOT}} with the root path of the project given by get_dir().

    Parameters
    ----------
    path : str
        The path to replace

    Returns
    -------
    new_path : str
        The new path with the root path replaced
    """
    root_path = get_dir()
    new_path = re.sub(r"{{BCNF_ROOT}}", root_path, path)

    return new_path


class ParameterIndexMapping:
    def __init__(self, parameters: list[str]) -> None:
        self.parameters = parameters
        self.map = {p: i for i, p in enumerate(parameters)}

    def __len__(self) -> int:
        return len(self.parameters)

    def vectorize(self, parameter_dict: dict) -> np.ndarray:
        return np.array([parameter_dict[p] for p in self.parameters]).T

    def dictify(self, parameter_vector: np.ndarray) -> dict:
        return {p: parameter_vector[i] for i, p in enumerate(self.parameters)}

    def __getitem__(self, key: str) -> int:
        return self.map[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.parameters)

    def __contains__(self, key: str) -> bool:
        return key in self.parameters

    def __repr__(self) -> str:
        return str(self.parameters)

    def __str__(self) -> str:
        return str(self.parameters)
