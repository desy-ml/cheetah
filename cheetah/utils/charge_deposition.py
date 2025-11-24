import torch

def deposit_charge_cic_2d(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bins1: torch.Tensor,
    bins2: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fast GPU-optimized Cloud-in-Cell (CIC) charge deposition in 2D.

    Parameters
    ----------
    x1 : (..., N)
        Particle x positions.
    x2 : (..., N)
        Particle y positions.

    bins1 : (Nx,)
        1D array of x-bin edges or centers (assumed uniform spacing).
    bins2 : (Ny,)
        1D array of y-bin edges or centers (assumed uniform spacing).

    weights : (..., N), optional
        Particle charge weights. If None, all particles have weight=1.

    Returns
    -------
    charge_grid : (..., Nx, Ny)
        Charge density on the grid.
    """

    device = x1.device
    dtype = x1.dtype

    # --- grid dimensions ---
    Nx = bins1.numel()
    Ny = bins2.numel()

    # --- infer dx, dy (assume uniform) ---
    dx = bins1[1] - bins1[0]
    dy = bins2[1] - bins2[0]

    if weights is None:
        weights = torch.ones_like(x1)

    # Expand bins to match batch dims
    # Normalize particle coordinates to grid index space
    #   u1, u2 represent positions in grid coordinates
    u1 = (x1 - bins1[0]) / dx
    u2 = (x2 - bins2[0]) / dy

    # Left cell index
    i1 = torch.floor(u1).to(torch.int64)
    i2 = torch.floor(u2).to(torch.int64)

    # Distances to right cell
    wx = u1 - i1
    wy = u2 - i2

    # 4 CIC weights (bilinear interpolation)
    w11 = (1 - wx) * (1 - wy)   # (i1,   i2)
    w21 = wx       * (1 - wy)   # (i1+1, i2)
    w12 = (1 - wx) * wy         # (i1,   i2+1)
    w22 = wx       * wy         # (i1+1, i2+1)

    # Prepare (i,j) index pairs for 4 corners
    # shape: (..., N)
    i1_clamp = i1.clamp(0, Nx - 1)
    i2_clamp = i2.clamp(0, Ny - 1)
    i1p      = (i1 + 1).clamp(0, Nx - 1)
    i2p      = (i2 + 1).clamp(0, Ny - 1)

    # ======= GPU-OPTIMIZED FLAT INDEXING ========
    # Convert 2D (i,j) → 1D index = i * Ny + j
    idx11 = i1_clamp * Ny + i2_clamp
    idx21 = i1p      * Ny + i2_clamp
    idx12 = i1_clamp * Ny + i2p
    idx22 = i1p      * Ny + i2p

    # Flatten batch dims and particle dim together
    batch_shape = x1.shape[:-1]
    B = int(torch.tensor(batch_shape).prod()) if batch_shape else 1
    N = x1.shape[-1]

    def flatten(t):
        return t.reshape(B, N)

    idx_all = torch.cat([
        flatten(idx11), flatten(idx21),
        flatten(idx12), flatten(idx22)
    ], dim=1)  # shape (B, 4N)

    vals_all = torch.cat([
        flatten(w11 * weights),
        flatten(w21 * weights),
        flatten(w12 * weights),
        flatten(w22 * weights),
    ], dim=1)  # shape (B, 4N)

    # Output buffer
    charge = torch.zeros((B, Nx * Ny), dtype=dtype, device=device)

    # Vectorized batched index_add_
    # Loop only over batch dimensions (tiny), not particles.
    for b in range(B):
        charge[b].index_add_(0, idx_all[b], vals_all[b])

    # Reshape back to original batch dims
    out_shape = (*batch_shape, Nx, Ny)
    return charge.reshape(out_shape)
