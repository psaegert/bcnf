import numpy as np
import torch
from tqdm import tqdm

from bcnf.models.cnf import CondRealNVP
from bcnf.simulation.physics import physics_ODE_simulation


def resimulate(model: CondRealNVP, T: int, dt: float, data_dict: dict[str, list], y_hat: torch.Tensor | None = None, X: torch.Tensor | None = None, m_samples: int = 1000, break_on_impact: bool = False, verbose: bool = True) -> np.ndarray:
    if y_hat is None:
        # Sample parameters from the model
        if X is None:
            raise ValueError("Either y_hat or X must be provided")
        y_hat = model.sample(n_samples=m_samples, y=X, verbose=verbose).cpu().numpy()  # (M, N, D)

    N = y_hat.shape[1]  # Number of simulations
    M = y_hat.shape[0]  # Number of parameter samples

    if verbose:
        print(f"Resimulating {N} trajectories {M} times")

    X_resimulation_list: list[list] = []

    for i in range(N):
        X_resimulation_list.append([])
        for j in tqdm(range(M), disable=not verbose, desc=f"Resimulating trajectory {i+1}/{N}"):
            X_resimulation_list[i].append(physics_ODE_simulation(
                T=T, dt=dt, break_on_impact=break_on_impact,
                **model.parameter_index_mapping.dictify(y_hat[j, i]),  # Learned parameters
                **{k: data_dict[k][i] for k in data_dict.keys() if k not in model.parameter_index_mapping.parameters})  # Fixed parameters
            )

    return np.array(X_resimulation_list)
