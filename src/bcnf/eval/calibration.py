import torch

from bcnf.models.cnf import CondRealNVP


def analyze_calibration(
        model: CondRealNVP,
        X: torch.Tensor,
        y: torch.Tensor,
        M_samples: int = 10_000,
        batch_size: int = 100,
        device: str = 'cpu',
        output_device: str = 'cpu',
        verbose: bool = True) -> torch.Tensor:
    model.to(device).eval()

    X = X.to(device)
    y = y.to(device)

    y_hat = model.sample(
        n_samples=M_samples,
        y=X,
        outer=True,
        batch_size=batch_size,
        output_device=output_device)

    y_hat_all = torch.cat([y_hat.to(output_device), y.unsqueeze(0).to(output_device)], dim=0)

    y_hat_all_sorted_ranks = torch.sum(y_hat_all < y.cpu().unsqueeze(0), dim=0)

    return y_hat_all_sorted_ranks
