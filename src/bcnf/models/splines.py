# https://github.com/bayesiains/nsf/blob/master/nde/transforms/splines/rational_quadratic.py

import numpy as np
import torch.nn.functional as F
import torch


class InputOutsideDomain(Exception):
    """Exception to be thrown when the input to a transform is not within its domain."""
    pass


def searchsorted(bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    bin_locations[:, -1] += eps
    return torch.sum(inputs[:, None] >= bin_locations, dim=-1) - 1


def unconstrained_rational_quadratic_spline(
        inputs: torch.Tensor,
        unnormalized_widths: torch.Tensor,
        unnormalized_heights: torch.Tensor,
        unnormalized_derivatives: torch.Tensor,
        inverse: bool = False,
        tails: str = 'linear',
        tail_bound: float = 1,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:

    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    log_det_J = torch.zeros_like(inputs)

    if tails == 'linear':
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[:, 0] = constant
        unnormalized_derivatives[:, -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        log_det_J[outside_interval_mask] = 0
    else:
        raise RuntimeError('{} tails are not implemented.'.format(tails))

    outputs[inside_interval_mask], log_det_J[inside_interval_mask] = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )

    return outputs, log_det_J


def normalize_bins(unnormalized_widths: torch.Tensor, min_bin_size: float, left: float, right: float) -> tuple[torch.Tensor, torch.Tensor]:
    n_bins = unnormalized_widths.shape[-1]

    # Normalize the widths
    widths = F.softmax(unnormalized_widths, dim=-1)

    # Scale the widths to be at least min_bin_width
    widths = min_bin_size + (1 - min_bin_size * n_bins) * widths

    # Compute the positions of the knots by integrating the widths
    cumulative_widths = torch.cumsum(widths, dim=-1)

    # Pad the cumulative widths with zeros?
    cumulative_widths = F.pad(cumulative_widths, pad=(1, 0), mode='constant', value=0.0)

    # Rescale the cumulative widths to the domain
    cumulative_widths = (right - left) * cumulative_widths + left
    cumulative_widths[:, 0] = left
    cumulative_widths[:, -1] = right

    # Recompute the widths from the cumulative widths
    widths = cumulative_widths[:, 1:] - cumulative_widths[:, :-1]

    return widths, cumulative_widths


def rational_quadratic_spline(
        inputs: torch.Tensor,
        unnormalized_widths: torch.Tensor,
        unnormalized_heights: torch.Tensor,
        unnormalized_derivatives: torch.Tensor,
        inverse: bool = False,
        left: float = 0,
        right: float = 1,
        bottom: float = 0,
        top: float = 1.,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:

    # Check that the inputs are within the domain
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise InputOutsideDomain()

    # Extract the number of bins from the list of unnormalized widths
    n_bins = unnormalized_widths.shape[-1]

    # Check that the minimal bin width and height are not too large
    if min_bin_width * n_bins > 1.0:
        raise ValueError(f'Minimal bin width too large for the number of bins. Got {min_bin_width} * {n_bins} = {min_bin_width * n_bins} > 1.0')
    if min_bin_height * n_bins > 1.0:
        raise ValueError(f'Minimal bin height too large for the number of bins. Got {min_bin_height} * {n_bins} = {min_bin_height * n_bins} > 1.0')

    # Normalize the widths & heights
    widths, cumulative_widths = normalize_bins(unnormalized_widths, min_bin_width, left, right)
    heights, cumulative_heights = normalize_bins(unnormalized_heights, min_bin_height, bottom, top)

    # Normalize the derivatives
    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    if inverse:
        bin_indices = searchsorted(bin_locations=cumulative_heights, inputs=inputs)[:, None]
    else:
        bin_indices = searchsorted(bin_locations=cumulative_widths, inputs=inputs)[:, None]

    input_cumwidths = cumulative_widths.gather(-1, bin_indices)[:, 0]
    input_bin_widths = widths.gather(-1, bin_indices)[:, 0]

    input_cumheights = cumulative_heights.gather(-1, bin_indices)[:, 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_indices)[:, 0]

    input_derivatives = derivatives.gather(-1, bin_indices)[:, 0]
    input_derivatives_plus_one = derivatives[:, 1:].gather(-1, bin_indices)[:, 0]

    input_heights = heights.gather(-1, bin_indices)[:, 0]

    if inverse:
        a = (((inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta) + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives - (inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * root.pow(2) + 2 * input_delta * theta_one_minus_theta + input_derivatives * (1 - root).pow(2))
        log_det_J = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -log_det_J
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2) + 2 * input_delta * theta_one_minus_theta + input_derivatives * (1 - theta).pow(2))
        log_det_J = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, log_det_J
