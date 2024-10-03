import torch


def elementwise_linspace(
    start: torch.Tensor, end: torch.Tensor, steps: int
) -> torch.Tensor:
    """
    Generate a tensor of linearly spaced values between two tensors element-wise.

    :param start: Any-dimensional tensor of the starting value for the set of points.
    :param end: Any-dimensional tensor of the ending value for the set of points.
    :param steps: Size of the last dimension of the constructed tensor.
    :return: A tensor of shape `start.shape + (steps,)` containing `steps` linearly
        spaced values between each pair of elements in `start` and `end`.
    """
    # Flatten the tensors
    a_flat = start.flatten()
    b_flat = end.flatten()

    # Create a list to store the results
    result = []

    # Generate linspace for each pair of elements in a and b
    for i in range(a_flat.shape[0]):
        result.append(torch.linspace(a_flat[i], b_flat[i], steps))

    # Stack the results along a new dimension (each linspace will become a row)
    result = torch.stack(result)

    # Reshape back to the original tensor dimensions with one extra dimension for the
    # steps
    new_shape = list(start.shape) + [steps]
    result = result.view(*new_shape)

    return result
