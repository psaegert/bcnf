import os
import pickle
import re
import warnings
from typing import Iterator

import numpy as np
import torch
from dynaconf import Dynaconf
from tqdm import tqdm


def load_config(config_file: str, verify: bool = True) -> dict:
    """
    Load a configuration file.

    Parameters
    ----------
    config_file : str
        The path to the configuration file.
    verify : bool, optional
        Whether to verify the configuration file, by default True.

    Returns
    -------
    config : dict
        The configuration dictionary.
    """

    if "{{BCNF_ROOT}}" not in config_file and verify:
        warnings.warn("The configuration file does not contain the placeholder '{{BCNF_ROOT}}'. This may cause issues when loading the model on a different machine.")

    config_file = sub_root_path(config_file)

    if not isinstance(config_file, str):
        raise TypeError("config_file must be a string.")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"File '{config_file}' does not exist.")

    config = Dynaconf(settings_files=[config_file])

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
        try:
            return np.array([parameter_dict[p] for p in self.parameters]).T
        except KeyError as e:
            raise KeyError(f'Parameter "{e}" not found in the parameter dictionary. Have available keys: {list(parameter_dict.keys())}')

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


def load_data(path: str, keep_output_type: str | None = None, n_files: int | None = None, verbose: bool = False, errors: str = 'raise') -> dict[str, list]:
    """
    Load data from a file or directory of files

    Parameters
    ----------
    path : str
        The path to the file or directory of files
    keep_output_type : str, optional
        The type of the output, by default None (all types are kept)
    n_files : int, optional
        The number of files to load, by default None (all files are loaded)
    verbose : bool, optional
        Whether to show a progress bar, by default False

    Returns
    -------
    dict
        A dictionary of the data
    """
    equivalent_keys = {
        'trajectories': ['traj', 'trajectory'],
        'videos': ['render', 'cams'],
    }

    if os.path.isfile(path):
        if verbose:
            print('Loading data from file...')
        with open(path, 'rb') as file:
            data = pickle.load(file)
    else:
        data = {}

        files = sorted(os.listdir(path))
        if n_files is not None:
            files = files[:n_files]

        pbar = tqdm(desc='Loading data from directory', disable=not verbose, total=len(files))

        for file in files:  # type: ignore
            pbar.set_postfix(file=file)
            # Read the file
            with open(os.path.join(path, file), 'rb') as f:  # type: ignore
                file_data = pickle.load(f)

                # Rename the keys to their canonical names
                for key, equivalent in equivalent_keys.items():
                    for e in equivalent:
                        if e in file_data:
                            file_data[key] = file_data.pop(e)

                # Add the data to the dictionary
                for key, value in file_data.items():
                    if key not in data:
                        data[key] = []
                    data[key].extend(value)

                del file_data
            pbar.update(1)

    # Keep only the specified output type (given that it is a key of equivalent_keys)
    if keep_output_type is not None and keep_output_type in equivalent_keys:
        for key in equivalent_keys.keys():

            # Remove every key that is not specified
            if key != keep_output_type and key in data:
                data.pop(key)

    # Check if all values have equal length
    value_lengths = [len(v) for v in data.values()]
    if len(set(value_lengths)) != 1:
        if errors == 'raise':
            raise ValueError(f'All values of the key "{key}" must have the same length')
        elif errors in ['print', 'warn']:
            print(f'Warning: All values of the key "{key}" must have the same length')
        elif errors == 'ignore':
            pass

    return data
