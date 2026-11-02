---
name: cheetah
description: >
  Differentiable particle accelerator beam dynamics simulation, lattice building,
  particle tracking, and gradient-based magnet optimization using Cheetah and PyTorch.
license: GPL-3.0
---

# Cheetah Accelerator Physics Skill

Cheetah is a high-speed, fully differentiable particle accelerator optics simulation engine written in PyTorch.

Use this skill when:
- Designing, modeling, or evaluating particle accelerator beamlines.
- Tracking particles (`ParticleBeam`) or distribution moments (`ParameterBeam`) through lattices (`Segment`).
- Matching Twiss parameters or optimizing magnet settings (quadrupoles, steerers, RF cavities) using PyTorch autograd gradients.
- Reading or writing lattices in LatticeJSON, Ocelot, Bmad, or Elegant formats.
- Running the Cheetah Model Context Protocol (MCP) server for automated beamline tuning.

---

## 1. Core Physics & Coordinate System

Cheetah uses a 7D vector $\vec{x} = (x, px, y, py, \tau, \delta, 1)$:
- $x, y$: Horizontal and vertical positions in meters ($m$).
- $px = P_x / p_0, py = P_y / p_0$: Normalized transverse momenta (dimensionless).
- $\tau = ct - s/\beta_0$: Longitudinal position relative to reference particle ($m$); $\tau < 0$ is bunch head.
- $\delta = (E - E_0) / (p_0 c)$: Relative energy deviation (dimensionless).
- $1$: Affine augmentation for thin-lens kicks and misalignments.
- **Reference Energy**: Always in electronvolts ($eV$).

---

## 2. Basic Beam Tracking Workflow

```python
import cheetah
import torch

# 1. Define lattice elements in a Segment
segment = cheetah.Segment(
    elements=[
        cheetah.Drift(length=torch.tensor(0.5)),
        cheetah.Quadrupole(
            length=torch.tensor(0.2), k1=torch.tensor(10.0), name="Q1"
        ),
        cheetah.Drift(length=torch.tensor(0.5)),
        cheetah.Screen(name="SCR1"),
    ]
)

# 2. Inspect lattice
print(segment.summary(as_string=True))

# 3. Create initial beam
beam = cheetah.ParameterBeam.from_twiss(
    beta_x=torch.tensor(3.0),
    beta_y=torch.tensor(5.0),
    alpha_x=torch.tensor(0.0),
    alpha_y=torch.tensor(0.0),
    emittance_x=torch.tensor(1e-6),
    emittance_y=torch.tensor(1e-6),
    energy=torch.tensor(1e8),  # 100 MeV in eV
)

# 4. Track beam through the lattice
outgoing_beam = segment.track(beam)
print(outgoing_beam.summary(as_string=True))
```

---

## 3. Differentiable Magnet Optimization (Beam Matching)

Because Cheetah is fully differentiable, you can use PyTorch optimizers (`torch.optim.Adam` or `torch.optim.LBFGS`) to optimize quadrupole strengths directly with analytical gradients:

```python
import cheetah
import torch

segment = cheetah.Segment(
    elements=[
        cheetah.Drift(length=torch.tensor(0.5)),
        cheetah.Quadrupole(length=torch.tensor(0.2), k1=torch.tensor(1.0), name="Q1"),
        cheetah.Drift(length=torch.tensor(0.5)),
        cheetah.Quadrupole(length=torch.tensor(0.2), k1=torch.tensor(-1.0), name="Q2"),
        cheetah.Drift(length=torch.tensor(0.5)),
    ]
)

# Make quadrupole strengths trainable parameters
segment.Q1.k1 = torch.nn.Parameter(segment.Q1.k1.clone())
segment.Q2.k1 = torch.nn.Parameter(segment.Q2.k1.clone())

optimizer = torch.optim.Adam([segment.Q1.k1, segment.Q2.k1], lr=0.1)
target_sigma_x = 1.5e-4

for step in range(50):
    optimizer.zero_grad()
    initial_beam = cheetah.ParameterBeam.from_twiss(
        beta_x=torch.tensor(2.0),
        beta_y=torch.tensor(2.0),
        energy=torch.tensor(1e8),
    )
    outgoing = segment.track(initial_beam)
    loss = (outgoing.sigma_x - target_sigma_x) ** 2
    loss.backward()
    optimizer.step()
```

---

## 4. Cheetah MCP Server Integration

Cheetah provides a built-in stdio MCP server for agent tool use:
- **Command**: `python -m cheetah.mcp`
- **Available Tools**:
  - `cheetah_inspect_lattice`: Inspects element list, types, cumulative $s$, and parameters.
  - `cheetah_track_beam`: Tracks a beam through a LatticeJSON lattice and returns `beam.summary()`.
  - `cheetah_match_beam`: Uses PyTorch autograd to tune specified quadrupoles to target beam sizes.
