# 3D Space Charge Simulations

Cheetah supports fast, differentiable 3D space charge simulations using the `SpaceChargeKick` element. This feature is particularly useful for simulating high-intensity beams where particle-particle interactions significantly impact the beam dynamics.

## Overview of the Method

Cheetah uses the **integrated Green's function method** to compute space charge effects:

1. **Grid Deposition**: The beam's charge density is deposited onto a 3D grid in the laboratory frame using the Cloud-in-Cell (CIC) charge deposition algorithm.
2. **Poisson Solver**: The Poisson equation is solved on the grid to find the electrostatic potential. To handle open boundaries, Cheetah uses the **Hockney method**, which doubles the grid size and performs convolution using Fast Fourier Transforms (FFTs).
3. **Field Computation**: The corresponding electromagnetic fields and Lorentz force components are computed on the grid.
4. **Particle Kick**: The forces are interpolated back to each macroparticle's position, and the particle momenta are updated.

Note that `SpaceChargeKick` only modifies the **momenta** (divergence and energy spread) of the beam and not the positions. It should be interleaved with elements that update the positions (like `Drift`).

## Prerequisites

Because space charge calculations rely on particle density distribution, this feature **only supports `ParticleBeam`** and cannot be used with `ParameterBeam`.

## Using SpaceChargeKick

To simulate space charge along a drift space, you can interleave `Drift` elements with `SpaceChargeKick` elements.

```python
import torch
import cheetah

# Define the drift length and the number of space charge steps
drift_length = 1.0
num_steps = 10
step_length = drift_length / num_steps

elements = []
for _ in range(num_steps):
    # Interleave a drift step and a space charge kick step
    elements.append(cheetah.Drift(length=torch.tensor(step_length)))
    elements.append(
        cheetah.SpaceChargeKick(
            effect_length=torch.tensor(step_length),
            grid_shape=(32, 32, 32),
            grid_extent_x=torch.tensor(
                3.0
            ),  # Grid width in x (multiples of beam sigma_x)
            grid_extent_y=torch.tensor(
                3.0
            ),  # Grid height in y (multiples of beam sigma_y)
            grid_extent_tau=torch.tensor(
                3.0
            ),  # Grid length in tau (multiples of beam sigma_tau)
        )
    )

segment = cheetah.Segment(elements=elements)
```

## Grid Configuration

The size and shape of the 3D grid can be configured using the following parameters:

- `grid_shape`: A tuple of three integers `(nx, ny, ntau)` specifying the grid resolution. Higher values increase accuracy but require more memory and computation.
- `grid_extent_x` / `grid_extent_y` / `grid_extent_tau`: The boundary extents of the grid, specified as multipliers of the beam's root-mean-square (RMS) size ($\sigma$) in the respective coordinate directions. For example, a value of `3.0` means the grid covers $\pm 3\sigma$ around the beam centre.
