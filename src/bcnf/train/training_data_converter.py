import warnings
from typing import Callable

import numpy as np
import torch


class RenderConverter:
    """
    Class to convert the input data for the feature network to the required format for training.

    Attributes
    ----------

    Methods
    -------
    get_converter(converter_function_name: str) -> Callable
        Factory method to get a converter function
    """
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_converter(converter_function_name: str) -> Callable:
        """
        Factory method to get a converter function.

        Parameters
        ----------
        converter_function_name : str
            Name of the converter function to use

        Returns
        -------
        converter : Callable
            The converter function
        """

        if converter_function_name == "":
            raise ValueError(f"Function {converter_function_name} does not exist")

        match converter_function_name:
            case "passthrough_tensor":
                return RenderConverter.passthrough_converter
            case "flatten_tensor":
                return RenderConverter.flatten_converter
            case _:
                warnings.warn(f"Converter '{converter_function_name}' not found. Will use default converter instead.")
                return RenderConverter.passthrough_converter

    @staticmethod
    def passthrough_converter(data: list[list[np.array]]) -> torch.Tensor:
        """
        Passthrough converter function

        Parameters
        ----------
        data : Any
            The input data to convert

        Returns
        -------
        converted_data : torch.Tensor
            The converted data
        """
        converted_data = torch.tensor(np.array(data))

        return converted_data

    @staticmethod
    def flatten_converter(data: list[list[np.array]]) -> torch.Tensor:
        """
        Flatten converter function

        Parameters
        ----------
        data : Any
            The input data to convert

        Returns
        -------
        converted_data : torch.Tensor
            The converted data
        """
        data_np = np.array(data)
        converted_data = torch.tensor(data_np.reshape(data_np.shape[0], -1))

        return converted_data
