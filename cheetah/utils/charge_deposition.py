import torch

def deposit_charge_cic_2d(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bins1: torch.Tensor,
    bins2: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Deposit particle charge onto a 2D grid using the Cloud-In-Cell (CIC) method,
    supporting arbitrary leading batch dimensions.

    Parameters
    ----------
    x1 : torch.Tensor, shape (..., N)
        Particle x-positions.

    x2 : torch.Tensor, shape (..., N)
        Particle y-positions.

    bins1 : torch.Tensor, shape (Nx,)
        Bin centers or edges along x.

    bins2 : torch.Tensor, shape (Ny,)
        Bin centers or edges along y.

    weights : torch.Tensor, shape (..., N)
        Particle weights (e.g., charge).

    Returns
    -------
    charge_grid : torch.Tensor, shape (..., Nx, Ny)
        Deposited charge density in the 2D grid.
    """

    device = x1.device
    dtype = x1.dtype

    batch_shape = x1.shape[:-1]  # all leading dims

    # ---------------------------------------------------------
    # Form meshgrid from bins
    # ---------------------------------------------------------
    grid_x, grid_y = torch.meshgrid(bins1, bins2, indexing='ij')  # (Nx, Ny)
    grid_shape = grid_x.shape  # (Nx, Ny)
    Nx, Ny = grid_shape

    # Grid origin (lower left corner)
    grid_origin = torch.stack([grid_x[0, 0], grid_y[0, 0]]).to(device=device, dtype=dtype)

    # Uniform cell size
    cell_size = torch.stack([
        grid_x[1, 0] - grid_x[0, 0],
        grid_y[0, 1] - grid_y[0, 0],
    ]).to(device=device, dtype=dtype)
    inv_cell_size = 1.0 / cell_size  # (2,)

    # ---------------------------------------------------------
    # Build stacked particle positions
    # ---------------------------------------------------------
    # (..., N, 2)
    particle_positions = torch.stack([x1, x2], dim=-1)

    # ---------------------------------------------------------
    # Normalize positions into cell coordinates
    # ---------------------------------------------------------
    # (..., N, 2)
    normalized = (particle_positions - grid_origin) * inv_cell_size

    # Lower-left cell integer index
    cell_idx = normalized.floor().to(torch.int)  # (..., N, 2)

    # 4 CIC neighbor offsets
    offsets = torch.tensor(
        [[0,0],[0,1],[1,0],[1,1]],
        device=device,
        dtype=torch.int
    )  # (4,2)

    # ---------------------------------------------------------
    # Surrounding cell indices
    # ---------------------------------------------------------
    # (..., N, 4, 2)
    neigh_idx = cell_idx.unsqueeze(-2) + offsets.unsqueeze(0).unsqueeze(0)

    # ---------------------------------------------------------
    # Compute CIC weights
    # ---------------------------------------------------------
    # (..., N, 4, 2)
    delta = normalized.unsqueeze(-2) - neigh_idx
    w = 1.0 - delta.abs()

    # (..., N, 4)
    cell_weights = w.prod(dim=-1)

    # ---------------------------------------------------------
    # Flatten everything for scatter_add_
    # ---------------------------------------------------------
    # weights: (..., N)
    # cell_weights: (..., N, 4)
    # values_flat: (..., 4*N)
    if weights is None:
        weights = torch.ones_like(x1)

    values_flat = (cell_weights * weights.unsqueeze(-1)).reshape(*weights.shape[:-1], -1)

    # neigh_idx: (..., N, 4, 2)
    idx = neigh_idx.reshape(*weights.shape[:-1], -1, 2)  # (..., 4*N, 2)
    idx_x = idx[..., 0]  # (..., 4*N)
    idx_y = idx[..., 1]  # (..., 4*N)

    # Valid mask inside grid range
    valid = (
        (idx_x >= 0) & (idx_x < grid_shape[0]) &
        (idx_y >= 0) & (idx_y < grid_shape[1])
    )

    # Apply mask while keeping batch dims
    idx_x = idx_x.masked_select(valid).reshape(*batch_shape, -1)
    idx_y = idx_y.masked_select(valid).reshape(*batch_shape, -1)
    values_flat = values_flat.masked_select(valid).reshape(*batch_shape, -1)

    # ---------------------------------------------------------
    # Convert 2D cell indices -> 1D flat index
    # ---------------------------------------------------------
    flat_index = idx_x * Ny + idx_y  # (..., K)

    # ---------------------------------------------------------
    # Allocate output grid
    # ---------------------------------------------------------
    charge_grid = torch.zeros(
        (*batch_shape, Nx * Ny),
        device=device,
        dtype=dtype
    )

    # ---------------------------------------------------------
    # Scatter-add into flat grid
    # ---------------------------------------------------------
    charge_grid.index_add_(dim=-1, index=flat_index, source=values_flat)

    # Reshape to (..., Nx, Ny)
    charge_grid = charge_grid.reshape(*batch_shape, Nx, Ny)

    # Normalize by cell area
    inv_area = inv_cell_size[0] * inv_cell_size[1]

    return charge_grid * inv_area
