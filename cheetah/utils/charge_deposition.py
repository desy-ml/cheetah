from typing import Sequence

import torch


def deposit_charge_cic(
    positions: Sequence[torch.Tensor],
    bins: Sequence[torch.Tensor],
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fast GPU-optimized Cloud-in-Cell (CIC) charge deposition in 1D, 2D, or 3D.

    :param positions: List or tuple of particle position tensors, each of shape
        `(..., N)`, leading with optional batch dimension.
        `N` is the number of particles.
        The length `d` of `positions` determines the dimensionality
        (1D, 2D, or 3D).
    :param bins: List or tuple of 1D tensors defining the grid coordinates for each
        dimension.
        ((N1,), (N2,), ...) where `Ni` is the number of grid points in dimension `i`.
        Each grid coordinate tensor must have uniform spacing.
        Length must match `positions`.
    :param weights: Particle charge weights of shape `(..., N)`.
        If `None`, all particles have weight 1.
    :return: Charge density on the d-dimensional grid with shape
        `(..., N1, N2, ..., Nd)`, where `d = len(positions)`.
    """
    # Validate inputs
    if not positions:
        raise ValueError("positions must contain at least one dimension")
    if len(positions) != len(bins):
        raise ValueError("positions and bins must have the same length")
    if len(positions) > 3:
        raise ValueError("Only 1D, 2D, and 3D CIC deposition are supported")

    ndim = len(positions)
    first_pos = positions[0]
    device = first_pos.device
    dtype = first_pos.dtype

    # Validate all position tensors have the same shape
    for i, pos in enumerate(positions[1:], 1):
        if pos.shape != first_pos.shape:
            raise ValueError(
                f"All position tensors must have the same shape. "
                f"positions[0] has shape {first_pos.shape}, "
                f"positions[{i}] has shape {pos.shape}"
            )
        if pos.device != device:
            raise ValueError("All tensors must be on the same device")
        if pos.dtype != dtype:
            raise ValueError("All tensors must have the same dtype")

    # Grid dimensions and spacing validation
    grid_sizes = []
    spacings = []

    for i, bin_array in enumerate(bins):
        N = bin_array.numel()
        if N < 2:
            raise ValueError(f"bins[{i}] must have at least 2 elements")

        spacing = bin_array[1] - bin_array[0]
        if N > 2:
            diffs = torch.diff(bin_array)
            if not torch.allclose(diffs, spacing, rtol=1e-4):
                raise ValueError(f"bins[{i}] must have uniform spacing")

        grid_sizes.append(N)
        spacings.append(spacing)

    if weights is None:
        weights = torch.ones_like(first_pos)

    # Set weights to zero for particles outside grid bounds
    for pos, bin_array in zip(positions, bins):
        outside_mask = (pos < bin_array[0]) | (pos >= bin_array[-1])
        weights = weights * (~outside_mask).float()

    # Normalize particle coordinates to grid index space
    grid_indices = []
    fractional_parts = []

    for pos, bin_array, spacing in zip(positions, bins, spacings):
        # Normalized coordinate in grid index space
        u = (pos - bin_array[0]) / spacing

        # Left cell index
        i = torch.floor(u).to(torch.int64)
        grid_indices.append(i)

        # Fractional distance to right cell
        w = u - i
        fractional_parts.append(w)

    # Generate all corner combinations and their weights
    num_corners = 2**ndim
    corner_indices = []
    corner_weights = []

    for corner in range(num_corners):
        # Determine which corners to use based on binary representation
        corner_offsets = []
        weight_factors = []

        for dim in range(ndim):
            if (corner >> dim) & 1:  # Use right cell in this dimension
                corner_offsets.append(1)
                weight_factors.append(fractional_parts[dim])
            else:  # Use left cell in this dimension
                corner_offsets.append(0)
                weight_factors.append(1 - fractional_parts[dim])

        # Calculate indices for this corner
        corner_idx_list = []
        for dim in range(ndim):
            base_idx = grid_indices[dim]
            offset_idx = (base_idx + corner_offsets[dim]).clamp(0, grid_sizes[dim] - 1)
            corner_idx_list.append(offset_idx)

        # Calculate weight for this corner
        corner_weight = weights
        for weight_factor in weight_factors:
            corner_weight = corner_weight * weight_factor

        corner_indices.append(corner_idx_list)
        corner_weights.append(corner_weight)

    # Convert multi-dimensional indices to flat indices
    def multi_to_flat_index(idx_list):
        flat_idx = idx_list[0]
        stride = 1
        for dim in range(1, ndim):
            stride *= grid_sizes[dim - 1]
            flat_idx = flat_idx + idx_list[dim] * stride
        return flat_idx

    # Flatten batch dims and particle dim together
    batch_shape = first_pos.shape[:-1]
    B = int(torch.tensor(batch_shape).prod()) if batch_shape else 1
    N = first_pos.shape[-1]

    def flatten_tensor(t):
        return t.reshape(B, N)

    # Prepare all indices and weights for batch processing
    all_flat_indices = []
    all_weights = []

    for corner_idx_list, corner_weight in zip(corner_indices, corner_weights):
        flat_idx = multi_to_flat_index(corner_idx_list)
        all_flat_indices.append(flatten_tensor(flat_idx))
        all_weights.append(flatten_tensor(corner_weight))

    # Concatenate all indices and weights
    idx_all = torch.cat(all_flat_indices, dim=1)  # shape (B, num_corners * N)
    vals_all = torch.cat(all_weights, dim=1)  # shape (B, num_corners * N)

    # Output buffer
    total_grid_size = int(torch.tensor(grid_sizes).prod())
    charge = torch.zeros((B, total_grid_size), dtype=dtype, device=device)

    # Vectorized batched index_add_
    for b in range(B):
        charge[b].index_add_(0, idx_all[b], vals_all[b])

    # Calculate inverse cell volume
    cell_volume = 1.0
    for spacing in spacings:
        cell_volume *= spacing
    inv_cell_volume = 1.0 / cell_volume
    charge = charge * inv_cell_volume

    # Reshape back to original batch dims + grid dims
    out_shape = (*batch_shape, *grid_sizes[::-1])
    charge = charge.reshape(out_shape)  # Grid dims are reversed by the reshape

    batch_ndim = len(batch_shape)
    spatial_axes = list(range(batch_ndim, batch_ndim + ndim))

    # Permute to put spatial axes in the correct order
    return charge.permute(*range(batch_ndim), *reversed(spatial_axes))


def deposit_charge_cic_1d(
    x: torch.Tensor,
    bins: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fast GPU-optimized Cloud-in-Cell (CIC) charge deposition in 1D.

    This is a convenience wrapper around `deposit_charge_cic` for 1D cases.
    Particles outside the grid bounds have their weights set to zero.

    :param x: Particle positions of shape `(..., N)`.
    :param bins: 1D tensor of grid coordinates with shape `(Nx,)`. Must have
        uniform spacing.
    :param weights: Particle charge weights of shape `(..., N)`. If `None`, all
        particles have weight 1.
    :return: Charge density on the 1D grid with shape `(..., Nx)`.
    """
    return deposit_charge_cic([x], [bins], weights)


def deposit_charge_cic_2d(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bins1: torch.Tensor,
    bins2: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fast GPU-optimized Cloud-in-Cell (CIC) charge deposition in 2D.

    This is a convenience wrapper around `deposit_charge_cic` for 2D cases.
    Particles outside the grid bounds have their weights set to zero.

    :param x1: Particle x positions of shape `(..., N)`.
    :param x2: Particle y positions of shape `(..., N)`.
    :param bins1: 1D tensor of x-grid coordinates with shape `(Nx,)`. Must have
        uniform spacing.
    :param bins2: 1D tensor of y-grid coordinates with shape `(Ny,)`. Must have
        uniform spacing.
    :param weights: Particle charge weights of shape `(..., N)`. If `None`, all
        particles have weight 1.
    :return: Charge density on the 2D grid with shape `(..., Nx, Ny)`.
    """
    return deposit_charge_cic(
        [x1, x2],
        [bins1, bins2],
        weights,
    )


def deposit_charge_cic_3d(
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    bins1: torch.Tensor,
    bins2: torch.Tensor,
    bins3: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fast GPU-optimized Cloud-in-Cell (CIC) charge deposition in 3D.

    This is a convenience wrapper around `deposit_charge_cic` for 3D cases.
    Particles outside the grid bounds have their weights set to zero.

    :param x1: Particle x positions of shape `(..., N)`.
    :param x2: Particle y positions of shape `(..., N)`.
    :param x3: Particle z positions of shape `(..., N)`.
    :param bins1: 1D tensor of x-grid coordinates with shape `(Nx,)`. Must have
        uniform spacing.
    :param bins2: 1D tensor of y-grid coordinates with shape `(Ny,)`. Must have
        uniform spacing.
    :param bins3: 1D tensor of z-grid coordinates with shape `(Nz,)`. Must have
        uniform spacing.
    :param weights: Particle charge weights of shape `(..., N)`. If `None`, all
        particles have weight 1.
    :return: Charge density on the 3D grid with shape `(..., Nx, Ny, Nz)`.
    """
    return deposit_charge_cic(
        [x1, x2, x3],
        [bins1, bins2, bins3],
        weights,
    )
