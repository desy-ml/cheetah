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
    tensors: list[torch.Tensor | None],
    desired_device: torch.device | None,
    desired_dtype: torch.dtype | None,
) -> tuple[torch.device, torch.dtype]:
    """
    Verifies that a unique device and dtype can be determined from the passed tensors
    and the optional desired device and dtype. If no desired values are requested,
    then all tensors (if they are not `None`) must have the same device and dtype.

    If all verifications pass, this function returns the determined device and dtype.
    """
    not_nones = [tensor for tensor in tensors if tensor is not None]

    chosen_device = (
        desired_device
        if desired_device is not None
        else are_all_the_same_device(not_nones)
    )
    chosen_dtype = (
        desired_dtype
        if desired_dtype is not None
        else are_all_the_same_dtype(not_nones)
    )
    return (chosen_device, chosen_dtype)
