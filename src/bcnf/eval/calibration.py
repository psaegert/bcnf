import numpy as np
import torch

from bcnf.models.cnf import CondRealNVP_v2


def CDF(sorted_arrray_indices: torch.Tensor, t: np.ndarray, M: int) -> np.ndarray:
    N = sorted_arrray_indices.shape[0]
    t = t * M
    return np.sum(sorted_arrray_indices[:, :, None] <= t[None, None, :], axis=0) / N


def brownian_confidence_interval(t: np.ndarray) -> np.ndarray:
    # Brownian bridge confidence interval
    # Usually scaled by 1/sqrt(N) to get the standard normal distribution
    # We instead scale the residuals by sqrt(N) in the compute_CDF_residuals function
    return np.sqrt(t * (1 - t))


def compute_y_hat_ranks(
        model: CondRealNVP_v2,
        y: torch.Tensor,
        *conditions: torch.Tensor,
        M_samples: int = 10_000,
        batch_size: int = 100,
        sample_batch_size: int | None = None,
        device: str = 'cpu',
        output_device: str = 'cpu',
        verbose: bool = True) -> torch.Tensor:
    if sample_batch_size is None:
        sample_batch_size = batch_size

    model.to(device).eval()

    y_hat = model.sample(
        M_samples,
        *conditions,
        outer=True,
        batch_size=batch_size,
        sample_batch_size=sample_batch_size,
        output_device=output_device,
        verbose=verbose)

    y_hat_all = torch.cat([y_hat.to(output_device), y.unsqueeze(0).to(output_device)], dim=0)

    y_hat_all_sorted_ranks = torch.sum(y_hat_all < y.cpu().unsqueeze(0), dim=0)

    return y_hat_all_sorted_ranks


def compute_CDF_residuals(
        y_hat_all_sorted_ranks: torch.Tensor,
        M_samples: int,
        t_divisions: int = 100,
        sigma: float = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    N_samples = y_hat_all_sorted_ranks.shape[0]

    t_linspace = np.linspace(0, 1, t_divisions)

    cdf = CDF(y_hat_all_sorted_ranks.cpu().numpy(), t_linspace, M_samples)

    # Compute the residuals of the actual CDF and the expected CDF
    residuals = cdf - t_linspace

    # Scale the residuals to compare them to the confidence intervals of Brownian Bridges
    scaled_residuals = residuals * np.sqrt(N_samples) / sigma

    confidence_interval = brownian_confidence_interval(t_linspace)

    return t_linspace, scaled_residuals, confidence_interval
