import math

import torch


def _kde_marginal_pdf(
    values: torch.Tensor,
    bins: torch.Tensor,
    sigma: torch.Tensor,
    weights: torch.Tensor | None = None,
    epsilon: torch.Tensor | float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the 1D marginal probability distribution function of the input tensor based
    on the number of histogram bins.

    :param values: Input tensor with shape :math:`(B, N)`. `B` is the vector shape.
    :param bins: Positions of the bins where KDE is computed.
        Shape :math:`(N_{bins})`.
    :param sigma: Gaussian smoothing factor with shape `(1,)`.
    :param weights: Input data weights of shape :math:`(B, N)`. Default to None.
        Use weights for heterogeneous sampling data.
    :param epsilon: A scalar, for numerical stability. Default: 1e-10.
    :return: Tuple of two tensors: (pdf, kernel_values).
        - pdf: Sum of the kernel values, gives an estimation of the marginal
        probability distribution function of shape :math:`(B, N_{bins})`.
        - kernel_values: Kernel values of all the input tensors of shape
        :math:`(B, N, N_{bins})`.
    """

    if not isinstance(values, torch.Tensor):
        raise TypeError(f"Input values type is not a torch.Tensor. Got {type(values)}")

    if not isinstance(bins, torch.Tensor):
        raise TypeError(f"Input bins type is not a torch.Tensor. Got {type(bins)}")

    if not isinstance(sigma, torch.Tensor):
        raise TypeError(f"Input sigma type is not a torch.Tensor. Got {type(sigma)}")

    if not bins.dim() == 1:
        raise ValueError(
            f"Input bins must be a of the shape NUM_BINS. Got {bins.shape}"
        )

    if not sigma.dim() == 0:
        raise ValueError(f"Input sigma must be a of the shape (1,). Got {sigma.shape}")

    values = values.unsqueeze(-1)

    if weights is None:
        weights = torch.ones_like(values)
    else:
        if not isinstance(weights, torch.Tensor):
            raise TypeError(f"Weights type is not a torch.Tensor. Got {type(weights)}")
        if weights.shape == values.shape[:-1]:
            weights = weights.unsqueeze(-1)
        if not weights.shape == values.shape:
            raise ValueError(
                f"Weights must have the same shape as values. Got {weights.shape}"
            )

    residuals = values - bins.repeat(*values.shape)
    kernel_values = (
        weights
        * torch.exp(-0.5 * (residuals / sigma).pow(2))
        / torch.sqrt(2 * math.pi * sigma**2)
    )

    prob_mass = torch.sum(kernel_values, dim=-2)
    normalization = torch.sum(prob_mass, dim=-1).unsqueeze(-1) + epsilon
    prob_mass = prob_mass / normalization

    return prob_mass, kernel_values


def _kde_joint_pdf_2d(
    kernel_values1: torch.Tensor,
    kernel_values2: torch.Tensor,
    epsilon: torch.Tensor | float = 1e-10,
) -> torch.Tensor:
    """
    Compute the joint probability distribution function of the input tensors based on
    the number of histogram bins.

    :param kernel_values1: shape :math:`(B, N, N_{bins})`.
    :param kernel_values2: shape :math:`(B, N, N_{bins})`.
    :param epsilon: A scalar, for numerical stability. Default: 1e-10.
    :return: Kernel density estimation of the joint probability distribution function of
        shape :math:`(B, N_{bins}, N_{bins})`.
    """

    if not isinstance(kernel_values1, torch.Tensor):
        raise TypeError(
            "Input kernel_values1 type is not a torch.Tensor."
            + f"Got {type(kernel_values1)}"
        )

    if not isinstance(kernel_values2, torch.Tensor):
        raise TypeError(
            "Input kernel_values2 type is not a torch.Tensor."
            + f"Got {type(kernel_values2)}"
        )

    joint_kernel_values = kernel_values1.transpose(-2, -1) @ kernel_values2
    normalization = (
        torch.sum(joint_kernel_values, dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
        + epsilon
    )
    pdf = joint_kernel_values / normalization

    return pdf


def kde_histogram_1d(
    x: torch.Tensor,
    bins: torch.Tensor,
    bandwidth: torch.Tensor,
    weights: torch.Tensor | None = None,
    epsilon: torch.Tensor | float = 1e-10,
) -> torch.Tensor:
    """
    Estimate the histogram using KDE of the input tensor.

    The computation uses kernel density estimation which requires a bandwidth
    (smoothing) parameter.

    :param x: Input tensor to compute the histogram with shape :math:`(B, D)`.
    :param bins: The number of bins to use the histogram :math:`(N_{bins})`.
    :param bandwidth: Gaussian smoothing factor with shape shape `(1,)`.
    :param weights: Weights of the input tensor of shape :math:`(B, N)`.
    :param epsilon: A scalar, for numerical stability. Default: 1e-10.
    :return: Computed 1d histogram of shape :math:`(B, N_{bins})`.

    Examples:
        >>> x = torch.rand(1, 10)
        >>> bins = torch.torch.linspace(0, 255, 128)
        >>> hist = kde_histogram_1d(x, bins, bandwidth=torch.tensor(0.9))
        >>> hist.shape
        torch.Size([1, 128])
    """

    pdf, _ = _kde_marginal_pdf(
        values=x,
        bins=bins,
        sigma=bandwidth,
        weights=weights,
        epsilon=epsilon,
    )

    return pdf


def kde_histogram_2d(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bins1: torch.Tensor,
    bins2: torch.Tensor,
    bandwidth: torch.Tensor,
    weights: torch.Tensor | None = None,
    epsilon: float | torch.Tensor = 1e-10,
) -> torch.Tensor:
    """
    Estimate the 2D histogram of the input tensor.

    The computation uses kernel density estimation which requires a bandwidth
    (smoothing) parameter.

    This is a modified version of the `kornia.enhance.histogram` implementation.

    :param x1: Input tensor to compute the histogram with shape :math:`(B, D1)`.
    :param x2: Input tensor to compute the histogram with shape :math:`(B, D2)`.
    :param bins: Bin coordinates.
    :param bandwidth: Gaussian smoothing factor with shape shape `(1,)`.
    :param weights: Weights of the input tensor of shape :math:`(B, N)`.
    :param epsilon: A scalar, for numerical stability. Default: 1e-10.
    :return: Computed histogram of shape :math:`(B, N_{bins}, N_{bins})`.

    Examples:
        >>> x1 = torch.rand(2, 32)
        >>> x2 = torch.rand(2, 32)
        >>> bins = torch.torch.linspace(0, 255, 128)
        >>> hist = kde_histogram_2d(x1, x2, bins, bandwidth=torch.tensor(0.9))
        >>> hist.shape
        torch.Size([2, 128, 128])
    """

    _, kernel_values1 = _kde_marginal_pdf(
        values=x1,
        bins=bins1,
        sigma=bandwidth,
        weights=weights,
    )
    _, kernel_values2 = _kde_marginal_pdf(
        values=x2,
        bins=bins2,
        sigma=bandwidth,
        weights=None,
    )  # Consider weights only one time

    joint_pdf = _kde_joint_pdf_2d(kernel_values1, kernel_values2, epsilon=epsilon)

    return joint_pdf
