import torch
from typing import Sequence


def deposit_charge_cic(
    positions: Sequence[torch.Tensor],
    bins: Sequence[torch.Tensor],
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fully vectorized Cloud-in-Cell (CIC) charge deposition in 1D, 2D, or 3D.
    
    High-performance GPU-optimized implementation where all operations 
    are fully vectorized using PyTorch tensor operations.

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
        
    Notes
    -----
    This implementation uses fully vectorized tensor operations for maximum 
    performance. Key optimizations include:
    - Precomputed corner offsets for all dimensions
    - Vectorized multi-dimensional to flat index conversion
    - Optimized memory access patterns with minimal intermediate tensors
    - Efficient batched grid accumulation
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
            raise ValueError(f"All position tensors must have the same shape. "
                           f"positions[0] has shape {first_pos.shape}, "
                           f"positions[{i}] has shape {pos.shape}")
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
        u = (pos - bin_array[0]) / spacing
        
        # Left cell index
        i = torch.floor(u).to(torch.int64)
        grid_indices.append(i)
        
        # Fractional distance to right cell
        w = u - i
        fractional_parts.append(w)

    # Precomputed corner offsets for different dimensions
    corner_offsets_dict = {
        1: torch.tensor([[0], [1]], device=device, dtype=torch.long),
        2: torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=device, dtype=torch.long),
        3: torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], device=device, dtype=torch.long)
    }
    
    if ndim not in corner_offsets_dict:
        raise ValueError(f"Precomputed offsets not available for {ndim}D. Only 1D, 2D, and 3D are supported.")
    
    corner_offsets = corner_offsets_dict[ndim]  # Shape: (num_corners, ndim)
    num_corners = corner_offsets.shape[0]
    
    # Stack grid indices and fractional parts for vectorized operations
    grid_indices_stacked = torch.stack(grid_indices, dim=-1)  # (..., N, ndim)
    fractional_parts_stacked = torch.stack(fractional_parts, dim=-1)  # (..., N, ndim)
    
    # Calculate all corner indices at once using broadcasting
    # grid_indices_stacked: (..., N, ndim) -> (..., N, 1, ndim)
    # corner_offsets: (num_corners, ndim) -> (1, 1, num_corners, ndim)
    corner_indices_all = (grid_indices_stacked.unsqueeze(-2) + 
                         corner_offsets.unsqueeze(-3))  # (..., N, num_corners, ndim)
    
    # Clamp to valid grid bounds
    grid_sizes_tensor = torch.tensor(grid_sizes, device=device, dtype=torch.long)
    corner_indices_all = corner_indices_all.clamp(
        min=torch.zeros_like(grid_sizes_tensor),
        max=grid_sizes_tensor - 1
    )
    
    # Calculate weights using vectorized operations
    # For each corner, weight = product over dims of (frac if offset=1, else 1-frac)
    # corner_offsets: (num_corners, ndim)
    # fractional_parts_stacked: (..., N, ndim) -> (..., N, 1, ndim)
    frac_expanded = fractional_parts_stacked.unsqueeze(-2)  # (..., N, 1, ndim)
    
    # Calculate weight factors for all corners at once
    # Where offset=1, use frac; where offset=0, use 1-frac
    weight_factors = torch.where(
        corner_offsets.bool(),  # (num_corners, ndim)
        frac_expanded,          # (..., N, 1, ndim) 
        1 - frac_expanded       # (..., N, 1, ndim)
    )  # (..., N, num_corners, ndim)
    
    # Product over dimensions to get final weights per corner
    corner_weights_all = weight_factors.prod(dim=-1)  # (..., N, num_corners)
    
    # Apply particle weights
    if weights is not None:
        corner_weights_all = corner_weights_all * weights.unsqueeze(-1)  # (..., N, num_corners)
    
    # FULLY VECTORIZED: Multi-dimensional to flat index conversion
    # Calculate strides for vectorized conversion
    strides = torch.ones(ndim, device=device, dtype=torch.long)
    for dim in range(1, ndim):
        strides[dim] = strides[dim-1] * grid_sizes[dim-1]
    
    # Vectorized flat index calculation for all corners at once
    # corner_indices_all shape: (..., N, num_corners, ndim)
    # strides shape: (ndim,)
    flat_indices_all = torch.sum(corner_indices_all * strides.unsqueeze(-2), dim=-1)  # (..., N, num_corners)
    
    # Prepare for batched operations
    batch_shape = first_pos.shape[:-1]
    B = int(torch.tensor(batch_shape).prod()) if batch_shape else 1
    N = first_pos.shape[-1]
    
    # Reshape to batch format: [B, N, num_corners]
    flat_indices_reshaped = flat_indices_all.reshape(B, N, num_corners)
    corner_weights_reshaped = corner_weights_all.reshape(B, N, num_corners)
    
    # Transpose and flatten for concatenation: [B, num_corners * N]
    idx_all = flat_indices_reshaped.transpose(1, 2).reshape(B, num_corners * N)
    vals_all = corner_weights_reshaped.transpose(1, 2).reshape(B, num_corners * N)

    # Output buffer
    total_grid_size = int(torch.tensor(grid_sizes).prod())
    charge = torch.zeros((B, total_grid_size), dtype=dtype, device=device)

    # Vectorized grid accumulation
    if B == 1:
        # Single batch - use simple index_add_
        charge[0].index_add_(0, idx_all[0], vals_all[0])
    else:
        # Multiple batches - vectorized approach
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, idx_all.shape[1])
        flat_batch_indices = batch_indices.reshape(-1)
        flat_grid_indices = idx_all.reshape(-1)
        flat_values = vals_all.reshape(-1)
        
        # Combined batch+grid indices for scatter_add
        combined_indices = flat_batch_indices * total_grid_size + flat_grid_indices
        charge_flat = charge.reshape(-1)
        charge_flat.index_add_(0, combined_indices, flat_values)
        charge = charge_flat.reshape(B, total_grid_size)

    # Reshape back to original batch dims + grid dims
    out_shape = (*batch_shape, *grid_sizes)

    # Calculate inverse cell volume
    cell_volume = 1.0
    for spacing in spacings:
        cell_volume *= spacing
    inv_cell_volume = 1.0 / cell_volume
    
    # Multiply by inverse cell volume to get charge density
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
    return deposit_charge_cic([x1, x2, x3], [bins1, bins2, bins3], weights)

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
    return deposit_charge_cic([x1, x2], [bins1, bins2], weights)
