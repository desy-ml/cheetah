from typing import Optional

import torch


def extract_argument_device(arguments: list[torch.Tensor]) -> torch.device:
    """
    Determines whether all arguments are on the same device and returns the default
    pytorch device if no argumente are passed.
    """
    if len(arguments) > 1:
        assert all(
            argument.device == arguments[0].device for argument in arguments
        ), "Arguments must be on the same device."

    return arguments[0].device if len(arguments) > 0 else torch.get_default_device()


def extract_argument_dtype(arguments: list[torch.Tensor]) -> torch.dtype:
    """
    Determines whether all arguments have the same dtype and returns the default
    pytorch dtype if no argumente are passed.
    """
    if len(arguments) > 1:
        assert all(
            argument.dtype == arguments[0].dtype for argument in arguments
        ), "Arguments must have the same dtype."

    return arguments[0].dtype if len(arguments) > 0 else torch.get_default_dtype()


def extract_argument_shape(arguments: list[torch.Tensor]) -> torch.Size:
    """Determines whether all arguments have the same shape."""
    if len(arguments) > 1:
        assert all(
            argument.shape == arguments[0].shape for argument in arguments
        ), "Arguments must have the same shape."

    return arguments[0].shape if len(arguments) > 0 else torch.Size([1])


def verify_device_and_dtype(
    required: list[torch.Tensor],
    optionals: list[Optional[torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.device, torch.dtype]:
    """
    Verifies that all required & given optional arguments have the same device and
    dtype if no defaults are provided.
    """
    not_nones = required + [argument for argument in optionals if argument is not None]

    device = device if device is not None else extract_argument_device(not_nones)
    dtype = dtype if dtype is not None else extract_argument_dtype(not_nones)
    return (device, dtype)
