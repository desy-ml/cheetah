# Virtual Diagnostics: Screens and BPMs

In particle accelerators, diagnostic devices are essential for monitoring beam positions, profiles, and transmission. Cheetah provides first-class simulation support for diagnostics via the `BPM` (Beam Position Monitor) and `Screen` (scintillating screen / OTR screen) elements.

Both elements are passive: by default, they act as zero-length elements that do not alter the beam, and they are skipped during tracking unless explicitly activated with `is_active=True`.

---

## Beam Position Monitors (`BPM`)

A `BPM` records the transverse centroid position $(x, y)$ of the bunch passing through it.

### Configuration and Usage

```python
import torch
import cheetah

# Define a BPM and activate recording
bpm = cheetah.BPM(
    name="BPM1",
    is_active=True,
    misalignment=torch.tensor([1e-4, -2e-4]),  # Optional (x, y) misalignment in meters
)

lattice = cheetah.Segment([bpm])

# Track any beam (ParameterBeam or ParticleBeam)
beam = cheetah.ParticleBeam.from_twiss(
    beta_x=torch.tensor(3.14),
    beta_y=torch.tensor(4.2),
    energy=torch.tensor(100e6),
    num_particles=10_000,
)
lattice.track(beam)

# Read the recorded centroid position [x, y] in meters
print("BPM reading (m):", bpm.reading)
```

### Properties
- **`bpm.reading`**: A 1D tensor containing the measured centroid `[x, y]` relative to the BPM's mechanical center.
- **`misalignment`**: Transverse offset `(x, y)` in meters. The reading reports `[mu_x - misalignment[0], mu_y - misalignment[1]]`.
- **Differentiability**: The reading is fully differentiable with respect to upstream lattice parameters (e.g. steerer strengths or quadrupole misalignments), making it suitable for orbit correction and reinforcement learning policies.

---

## Scintillating & OTR Screens (`Screen`)

A `Screen` simulates a camera viewing a scintillator or optical transition radiation (OTR) screen, producing a 2D digital image (PyTorch tensor) of the beam distribution.

```python
import matplotlib.pyplot as plt
import torch
import cheetah

# Configure a diagnostic camera screen
screen = cheetah.Screen(
    name="SCREEN1",
    resolution=(256, 256),  # Camera sensor (width, height) in pixels
    pixel_size=torch.tensor([50e-6, 50e-6]),  # Physical pixel size in meters (50 µm)
    binning=1,  # Camera binning factor (integer)
    method="cloud-in-cell",  # Image generation method
    is_active=True,  # Enable recording during tracking
    is_blocking=False,  # If True, stops the beam downstream
)

lattice = cheetah.Segment([screen])

# Track through the screen
lattice.track(beam)

# Access the simulated 2D camera image
image = screen.reading  # Tensor of shape (height, width)

# Display the simulated screen image
plt.figure(figsize=(6.0, 5.0))
plt.imshow(
    image.detach().cpu(),
    extent=[
        -screen.extent[0].item() * 1e3,
        screen.extent[1].item() * 1e3,
        -screen.extent[2].item() * 1e3,
        screen.extent[3].item() * 1e3,
    ],
    cmap="viridis",
    origin="lower",
)
plt.colorbar(label="Intensity")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.title("Simulated Screen Image")
```

---

## Image Generation Methods

Cheetah supports three methods for generating the 2D image from a `ParticleBeam`, balancing computational speed against differentiability:

| Method | Relative Speed | Differentiable? | Vectorisation | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **`"cloud-in-cell"`** (default) | Fast (~1.5× slower than histogram) | **Yes** (smooth gradients) | Yes | Gradient-based tuning, profile matching, surrogate training |
| **`"histogram"`** | Fastest | No (zero gradients almost everywhere) | No | Fast forward sweeps, non-differentiable evaluations |
| **`"kde"`** | Slow (~280× slower than histogram) | **Yes** (smooth gradients) | Yes | High-resolution analytic smoothing with custom `kde_bandwidth` |

> [!NOTE]
> When tracking a `ParameterBeam`, `Screen` automatically computes the analytic 2D multivariate Gaussian probability density directly on the pixel grid.

---

## Screen Parameters & Misalignments

- **`resolution`**: Sensor resolution `(width, height)` in pixels.
- **`pixel_size`**: Size of a pixel on the screen in meters `(width, height)`.
- **`binning`**: Integer binning factor; reduces effective camera resolution and increases pixel size accordingly.
- **`misalignment`**: Offset `torch.tensor([dx, dy])` in meters of the screen center relative to the beam axis.
- **`is_blocking`**: When set to `True`, the screen absorbs the beam, stopping particles from continuing downstream.
