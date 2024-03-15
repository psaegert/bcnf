from typing import Any

import torch
from torchsummary import summary

from bcnf.models import CondRealNVP, FullyConnectedFeatureNetwork

# from torchviz import make_dot


class TrainerModelHandler:
    def __init__(self) -> None:
        pass

    def make_model(self,
                   config: dict,
                   data_size_primary: torch.Size,
                   data_size_feature: torch.Size,
                   device: torch.device,
                   data_type: Any = torch.float32) -> torch.nn.Module:
        """
        Create the model for training

        Parameters
        ----------
        config : dict
            A dictionary containing the configuration for the model
        device : torch.device
            The device to use for training

        Returns
        -------
        model : torch.nn.Module
            The model for training
        """
        print("Creating the model...")
        data_size_primary_int = data_size_primary[0]
        data_size_feature_int = data_size_feature[0]

        feature_network_sized = list(config["feature_network"]["hidden_layer_sizes"])
        feature_network_sized.append(config["feature_network"]["n_conditions"])
        feature_network_sized.insert(0, data_size_feature_int)

        feature_network = FullyConnectedFeatureNetwork(sizes=feature_network_sized,
                                                       dropout=config["dropout"],
                                                       ).to(data_type).to(device)

        # Create the model
        model = CondRealNVP(size=data_size_primary_int,
                            nested_sizes=config["nested_sizes"],
                            n_blocks=config["n_blocks"],
                            n_conditions=config["feature_network"]["n_conditions"],
                            feature_network=feature_network,
                            dropout=config["dropout"],
                            act_norm=config["act_norm"],
                            device=device).to(data_type).to(device)

        return model

    @staticmethod
    def inn_nll_loss(z: torch.Tensor,
                     log_det_J: torch.Tensor,
                     reduction: str = 'mean') -> torch.Tensor:
        """
        Compute the negative log-likelihood loss for the INN

        Parameters
        ----------
        z : torch.Tensor
            The input tensor
        log_det_J : torch.Tensor
            The log determinant of the Jacobian
        reduction : str
            The type of reduction to use. The default is 'mean'

        Returns
        -------
        loss : torch.Tensor
            The negative log-likelihood loss
        """
        if reduction == 'mean':
            return torch.mean(0.5 * torch.sum(z**2, dim=1) - log_det_J)
        else:
            return 0.5 * torch.sum(z**2, dim=1) - log_det_J

    def verify_model(self,
                     model: torch.nn.Module,
                     test_data: Any) -> None:
        # Verify the model
        # Option 1
        summary(model, input_size=[test_data[1].shape, test_data[0].shape])

        # Option 2 TODO: Fix this
        '''
        x = dataset[0][1].to(self.device)
        y = dataset[0][0].to(self.device)

        print("Input sizes for the model:")
        print("Input to primary NF:", x.shape, x.dtype)
        print("Input to conditioning network:", y.shape, y.dtype)

        output = model.forward(x, y)
        output = output.to("cpu")
        model = model.to("cpu")
        graph = make_dot(output, params=dict(model.named_parameters()))
        # Save the graph visualization as an image
        graph.render(filename="computation_graph", directory="./", format="png")
        # print(graph)
        '''
