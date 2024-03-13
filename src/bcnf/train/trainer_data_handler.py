import pickle

import torch
from torch.utils.data import TensorDataset

from bcnf.simulation.sampling import generate_data


class TrainerDataHandler:
    def __init__(self) -> None:
        pass

    def get_data_for_training(self, config: dict) -> torch.utils.data.TensorDataset:
        """
        Gts data for training the model

        Parameters
        ----------
        config : dict
            A dictionary containing the configuration parameters for the data

        Returns
        -------
        dataset : torch.utils.data.TensorDataset
            A PyTorch TensorDataset containing the data for training the model
        """
        if (config["load"]):
            print(f"Loading data from {config['load_path']}")
            X, y = self._load_data_for_training(config["load_path"])
        else:
            print("Generating data")
            X, y = self._generate_data_for_training(config)

        # Matches pairs of lables and data, so dataset[0] returns tuple of the first entry in X and y
        dataset = TensorDataset(X, y)

        return dataset

    def _load_data_for_training(self, filename: str) -> tuple[torch.tensor, torch.tensor]:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Data for feature network
        if 'cams' in data:
            X = data['cams']
        elif 'trj' in data:
            X = data['trj']
        else:
            X = torch.zeros(len(data['x0_x']))

        # Data for primary network
        keys = list(data.keys())
        keys = [key for key in keys if key not in ['cams', 'trj']]
        # Convert lists to tensors
        tensors = [torch.tensor(data[key]) for key in keys]
        # Stack tensors along a new dimension
        y = torch.stack(tensors, dim=1)

        return X, y

    def _generate_data_for_training(self, config: dict) -> tuple[torch.tensor, torch.tensor]:
        X, y = generate_data(name=config["name"],
                             overwrite=config["overwrite"],
                             config_file=config["data_generation_config_file"],
                             n=config["n_samples"],
                             type=config["data_type"],
                             SPF=config["SPF"],
                             T=config["T"],
                             ratio=config["ratio"],
                             fov_horizontal=config["fov_horizontal"],
                             cam1_pos=config["cam1_pos"],
                             print_acc_rej=config["print_acc_rej"],
                             num_cams=config["num_cams"],
                             break_on_impact=config["break_on_impact"],
                             verbose=config["verbose"])

        return X, y

    def make_data_loader(dataset: torch.tensor,
                         batch_size: int,
                         num_workers: int = 2) -> torch.utils.data.DataLoader:
        """
        Create a DataLoader for the given dataset

        Parameters
        ----------
        dataset : torch.tensor
            The dataset to use for training
        batch_size : int
            The batch size to use for training
        num_workers : int
            The number of workers to use for loading the data

        Returns
        -------
        loader : torch.utils.data.DataLoader
            The DataLoader for the given dataset
        """
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=num_workers)
        return loader
