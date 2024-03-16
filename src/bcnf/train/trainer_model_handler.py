# from typing import Any

# import torch

# from bcnf.models import CondRealNVP, FullyConnectedFeatureNetwork

# # from torchviz import make_dot


# class TrainerModelHandler:
#     def __init__(self) -> None:
#         pass

#     def make_model(self,
#                    config: dict,
#                    data_size_primary: torch.Size,
#                    data_size_feature: torch.Size,
#                    device: torch.device,
#                    data_type: Any = torch.float32) -> torch.nn.Module:
#         """
#         Create the model for training

#         Parameters
#         ----------
#         config : dict
#             A dictionary containing the configuration for the model
#         device : torch.device
#             The device to use for training

#         Returns
#         -------
#         model : torch.nn.Module
#             The model for training
#         """
#         print("Creating the model...")
#         data_size_primary_int = data_size_primary[0]
#         data_size_feature_int = data_size_feature[0]

#         feature_network_sized = list(config["feature_network"]["hidden_layer_sizes"])
#         feature_network_sized.append(config["feature_network"]["n_conditions"])
#         feature_network_sized.insert(0, data_size_feature_int)

#         feature_network = FullyConnectedFeatureNetwork(sizes=feature_network_sized,
#                                                        dropout=config["dropout"],
#                                                        ).to(data_type).to(device)

#         # Create the model
#         model = CondRealNVP(size=data_size_primary_int,
#                             nested_sizes=config["nested_sizes"],
#                             n_blocks=config["n_blocks"],
#                             n_conditions=config["feature_network"]["n_conditions"],
#                             feature_network=feature_network,
#                             dropout=config["dropout"],
#                             act_norm=config["act_norm"],
#                             device=device).to(data_type).to(device)

#         return model
