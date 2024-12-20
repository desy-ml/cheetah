from typing import Optional, Union

import torch


def are_all_the_same_device(tensors: list[torch.Tensor]) -> torch.device:
    """
    Determines whether all arguments are on the same device and, if so, returns that
    device. If no arguments are passed, global default PyTorch device is returned.
    """
    if len(tensors) > 1:
        assert all(
            argument.device == tensors[0].device for argument in tensors
        ), "All tensors must be on the same device."

    return tensors[0].device if len(tensors) > 0 else torch.get_default_device()


def are_all_the_same_dtype(tensors: list[torch.Tensor]) -> torch.dtype:
    """
    Determines whether all arguments have the same dtype and, if so, returns that dtype.
    If no arguments are passed, global default PyTorch dtype is returned.
    """
    if len(tensors) > 1:
        assert all(
            argument.dtype == tensors[0].dtype for argument in tensors
        ), "All arguments must have the same dtype."

    return tensors[0].dtype if len(tensors) > 0 else torch.get_default_dtype()


def verify_device_and_dtype(
    arguments: list[Optional[Union[torch.Tensor, float]]],
    desired_device: Optional[torch.device],
    desired_dtype: Optional[torch.dtype],
) -> tuple[torch.device, torch.dtype]:
    """
    Verifies that a unique device and dtype can be determined from the passed arguments
    and the optional desired device and dtype. If no desired values are requested,
    then all tensors must have the same device and dtype.

    If all verifications pass, this function returns the determined device and dtype.
    """
    tensors = [arg for arg in arguments if isinstance(arg, torch.Tensor)]

    chosen_device = (
        desired_device
        if desired_device is not None
        else are_all_the_same_device(tensors)
    )
    chosen_dtype = (
        desired_dtype if desired_dtype is not None else are_all_the_same_dtype(tensors)
    )
    return (chosen_device, chosen_dtype)
