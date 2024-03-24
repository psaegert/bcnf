import concurrent.futures

import numpy as np
import torch
from tqdm import tqdm

from bcnf.models.cnf import CondRealNVP_v2
from bcnf.simulation.physics import physics_ODE_simulation
from bcnf.utils import ParameterIndexMapping


def resimulate_trajectory(args: tuple[int, int, float, bool, torch.Tensor, dict[str, float], ParameterIndexMapping]) -> np.ndarray:
    j, T, dt, break_on_impact, y_hat_i, fixed_parameters, parameter_index_mapping = args
    return physics_ODE_simulation(
        T=T, dt=dt, break_on_impact=break_on_impact,
        **parameter_index_mapping.dictify(y_hat_i),  # Learned parameters
        **fixed_parameters  # Fixed parameters
    )


def resimulate(
        model: CondRealNVP_v2,
        T: int,
        dt: float,
        data_dict: dict[str, list],
        y_hat: torch.Tensor | None = None,
        *conditions: torch.Tensor,
        m_samples: int = 1000,
        break_on_impact: bool = False,
        n_procs: int = None,
        batch_size: int = 100,
        verbose: bool = True) -> np.ndarray:

    if y_hat is None:
        if len(conditions) != model.feature_network_stack.n_distinct_conditions:
            raise ValueError(f"Expected {model.feature_network_stack.n_distinct_conditions} conditions, got {len(conditions)}")
        y_hat = model.sample(m_samples, *conditions, batch_size=batch_size, verbose=verbose).cpu().numpy()  # (M, N, D)

    N = y_hat.shape[1]  # Number of simulations
    M = y_hat.shape[0]  # Number of parameter samples

    if verbose:
        print(f"Resimulating {N} trajectories {M} times")

    X_resimulation_list: list[list] = []

    pbar = tqdm(total=N, desc=f"Resimulating trajectories with {n_procs} processes", disable=not verbose)

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as executor:
        for i in range(N):
            X_resimulation_list.append([])
            args_list = [(j, T, dt, break_on_impact, y_hat[j, i], {k: data_dict[k][i] for k in data_dict.keys() if k not in model.parameter_index_mapping.parameters}, model.parameter_index_mapping) for j in range(M)]

            results = executor.map(resimulate_trajectory, args_list)

            X_resimulation_list[i].extend(results)
            pbar.update(1)

    return np.array(X_resimulation_list)
