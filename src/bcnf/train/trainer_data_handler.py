import pickle

import numpy as np
import torch
from torch._C import dtype as torch_dtype
from torch.utils.data import DataLoader, RandomSampler, Subset, TensorDataset

from bcnf.simulation.sampling import generate_data
from bcnf.train.training_data_converter import RenderConverter


class TrainerDataHandler:
    def __init__(self) -> None:
        pass

    def get_data_for_training(
            self,
            config: dict,
            dtype: torch_dtype) -> TensorDataset:
        """
        Gts data for training the model

        Parameters
        ----------
        config : dict
            A dictionary containing the configuration parameters for the data

        Returns
        -------
        dataset : TensorDataset
            A PyTorch TensorDataset containing the data for training the model
        """
        if (config["load"]):
            print(f"Loading data from {config['load_path']}")
            data = self._load_data_for_training(config["load_path"])
        else:
            print("Generating data")
            data = self._generate_data_for_training(config)

        conversion_function = RenderConverter.get_converter(config["converter"])

        # Data for feature network
        if 'cams' in data:
            X = conversion_function(data['cams'])
        elif 'trajectory' in data:
            X = data['trajectory']
        else:
            X = torch.zeros(len(data['x0_x']))

        # Data for primary network -> make it n x #parameters
        keys = list(data.keys())
        keys = [key for key in keys if key not in ['cams', 'trajectory']]
        tensors = [torch.tensor(np.array(data[key])) for key in keys]
        # Split nxx tensors into x separate tensors
        split_tensors = [torch.split(tensor, 1, dim=1) if tensor.dim() == 2 else [tensor] for tensor in tensors]
        # Flatten the list of lists
        split_tensors = [item for sublist in split_tensors for item in sublist]
        # Check if any tensors have shape nx1 and squeeze them to n
        split_tensors = [tensor.squeeze() if len(tensor.shape) > 1 else tensor for tensor in split_tensors]
        # Stack all tensors along dimension 1
        y = torch.stack(split_tensors, dim=1)

        # Make the correct type for the data
        X = X.to(dtype)
        y = y.to(dtype)

        # Matches pairs of lables and data, so dataset[0] returns tuple of the first entry in X and y
        dataset = TensorDataset(X, y)

        return dataset

    def _load_data_for_training(self, filename: str) -> dict[str, list]:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        return data

    def _generate_data_for_training(self, config: dict) -> dict[str, list]:
        data = generate_data(
            name=config["name"],
            overwrite=config["overwrite"],
            config_file=config["data_generation_config_file"],
            n=config["n_samples"],
            type=config["data_type"],
            SPF=config["dt"],
            T=config["T"],
            ratio=config["ratio"],
            fov_horizontal=config["fov_horizontal"],
            print_acc_rej=config["print_acc_rej"],
            num_cams=config["num_cams"],
            break_on_impact=config["break_on_impact"],
            verbose=config["verbose"])

        return data

    def make_data_loader(
            self,
            dataset: TensorDataset,
            batch_size: int,
            pin_memory: bool,
            num_workers: int = 2) -> DataLoader:
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
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers)

    def verify_data(
            self,
            data: DataLoader) -> None:
        """
        Verify the data

        Parameters
        ----------
        data : torch.utils.data.DataLoader
            The data to verify

        Returns
        -------
        None
        """
        print()
        print("Verifying data:")
        print("Feature network input shape:", data[0][0].shape)
        print("Feature network device:", data[0][0].device)
        print("NF network input shape:", data[0][1].shape)
        print("NF network device:", data[0][1].device)

    def split_dataset(
            self,
            dataset: DataLoader,
            split_ratio: float) -> tuple[Subset, Subset]:
        """
        Split the data into training and validation sets

        Parameters
        ----------
        data : torch.utils.data.DataLoader
            The data to split
        split_ratio : float
            The ratio to split the data

        Returns
        -------
        train_data : torch.utils.data.Subset
            The training data
        val_data : torch.utils.data.Subset
            The validation data
        """
        train_size = int(split_ratio * len(dataset))

        # Create a random sampler to shuffle the indices
        indices = list(range(len(dataset)))
        RandomSampler(indices)

        # Split indices into training and validation sets
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Create Subset datasets
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        return train_dataset, val_dataset
