import torch
from torch.utils.data import TensorDataset

from bcnf.simulation.physics import get_data


class TrainerDataHandler:
    def __init__(self) -> None:
        pass

    def generate_data_for_training(self, config: dict) -> torch.utils.data.TensorDataset:
        """
        Generate data for training the model

        Parameters
        ----------
        config : dict
            A dictionary containing the configuration parameters for the data generation

        Returns
        -------
        dataset : torch.utils.data.TensorDataset
            A PyTorch TensorDataset containing the data for training the model"""
        X, y = get_data(T=config["T"],
                        dt=config["dt"],
                        N=config["N"],
                        break_on_impact=config["break_on_impact"])

        X_tensor = torch.Tensor(X.reshape(X.shape[0], -1))
        y_tensor = torch.Tensor(y)

        # Matches pairs of lables and data, so dataset[0] returns tuple of the first entry in X and y
        dataset = TensorDataset(X_tensor, y_tensor)

        return dataset

    def load_data_for_training(self, filename: str) -> torch.utils.data.TensorDataset:
        raise NotImplementedError("This method has not been implemented yet.")

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
