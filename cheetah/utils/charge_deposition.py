from typing import Sequence

import torch


def deposit_charge_cic(
    positions: Sequence[torch.Tensor],
    bins: Sequence[torch.Tensor],
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fast GPU-optimized Cloud-in-Cell (CIC) charge deposition in 1D, 2D, or 3D.

    Parameters
    ----------
    positions : sequence of torch.Tensor
        List/tuple of particle position tensors, each of shape (..., N).
        Length determines dimensionality (1D, 2D, or 3D).
    bins : sequence of torch.Tensor
        List/tuple of 1D arrays defining bin edges for each dimension.
        Must have uniform spacing. Length must match positions.
    weights : (..., N), optional
        Particle charge weights. If None, all particles have weight=1.

    Returns
    -------
    charge_grid : (..., N1, N2, ..., Nd)
        Charge density on the d-dimensional grid, where d is len(positions).
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
    normalized_coords = []
    grid_indices = []
    fractional_parts = []

    for pos, bin_array, spacing in zip(positions, bins, spacings):
        u = (pos - bin_array[0]) / spacing
        normalized_coords.append(u)

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

    # Reshape back to original batch dims + grid dims
    out_shape = (*batch_shape, *grid_sizes[::-1])

    # Calculate inverse cell volume
    cell_volume = 1.0
    for spacing in spacings:
        cell_volume *= spacing
    inv_cell_volume = 1.0 / cell_volume
    charge = charge * inv_cell_volume

    return charge.reshape(out_shape)


def deposit_charge_cic_1d(
    x: torch.Tensor,
    bins: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fast GPU-optimized Cloud-in-Cell (CIC) charge deposition in 1D.

    This is a convenience wrapper around deposit_charge_cic for 1D cases.
    Particles outside the grid bounds have their weights set to zero.

    Parameters
    ----------
    x : (..., N)
        Particle positions.
    bins : (Nx,)
        1D array of bin edges or centers (must have uniform spacing).
    weights : (..., N), optional
        Particle charge weights. If None, all particles have weight=1.

    Returns
    -------
    charge_grid : (..., Nx)
        Charge density on the 1D grid.
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

    This is a convenience wrapper around deposit_charge_cic for 2D cases.
    Particles outside the grid bounds have their weights set to zero.

    Parameters
    ----------
    x1 : (..., N)
        Particle x positions.
    x2 : (..., N)
        Particle y positions.
    bins1 : (Nx,)
        1D array of x-bin edges or centers (must have uniform spacing).
    bins2 : (Ny,)
        1D array of y-bin edges or centers (must have uniform spacing).
    weights : (..., N), optional
        Particle charge weights. If None, all particles have weight=1.

    Returns
    -------
    charge_grid : (..., Nx, Ny)
        Charge density on the 2D grid.
    """
    charge_density = deposit_charge_cic(
        [x1, x2],
        [bins1, bins2],
        weights,
    )

    return charge_density.mT


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

    This is a convenience wrapper around deposit_charge_cic for 3D cases.
    Particles outside the grid bounds have their weights set to zero.

    Parameters
    ----------
    x1 : (..., N)
        Particle x positions.
    x2 : (..., N)
        Particle y positions.
    x3 : (..., N)
        Particle z positions.
    bins1 : (Nx,)
        1D array of x-bin edges or centers (must have uniform spacing).
    bins2 : (Ny,)
        1D array of y-bin edges or centers (must have uniform spacing).
    bins3 : (Nz,)
        1D array of z-bin edges or centers (must have uniform spacing).
    weights : (..., N), optional
        Particle charge weights. If None, all particles have weight=1.

    Returns
    -------
    charge_grid : (..., Nx, Ny, Nz)
        Charge density on the 3D grid.
    """
    charge_density = deposit_charge_cic(
        [x1, x2, x3],
        [bins1, bins2, bins3],
        weights,
    )

    return charge_density.mT
    return charge_density.mT
