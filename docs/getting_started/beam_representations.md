# Beam Representations in Cheetah

Cheetah provides two distinct beam classes for tracking charged particle bunches: `ParameterBeam` and `ParticleBeam`. Choosing the right beam representation allows you to balance physical fidelity against computational speed and memory consumption.

---

## Comparison: `ParameterBeam` vs. `ParticleBeam`

| Feature                         | `ParameterBeam`                                                    | `ParticleBeam`                                                           |
| :------------------------------ | :----------------------------------------------------------------- | :----------------------------------------------------------------------- |
| **Data Representation**         | 7D centroid vector $\mu$ & $7 \times 7$ covariance matrix $\Sigma$ | $N \times 7$ tensor of individual macroparticles                         |
| **Computational Complexity**    | $O(1)$ constant time ($7 \times 7$ matrix multiplications)         | $O(N)$ linear in particle count (or $O(N + M \log M)$ with space charge) |
| **Memory Footprint**            | Extremely low (~few hundred bytes)                                 | Proportional to macroparticle count $N$                                  |
| **Physical Scope**              | Linear transfer maps and 2nd-order aberrations                     | All maps, nonlinear apertures, and 3D space charge kicks                 |
| **Particle Loss & Collimation** | Not modeled                                                        | Tracks lost particles through `Aperture` elements                        |
| **Differentiability**           | Fully differentiable (`autograd`)                                  | Fully differentiable (`autograd`)                                        |
| **Primary Use Cases**           | Fast envelope matching, Bayesian optimisation, online tuning       | High-intensity space charge, realistic phase space distributions         |

---

## `ParameterBeam`

A `ParameterBeam` models the beam ensemble through its statistical moments: the 7-dimensional mean coordinate vector $\mu$ and the $7 \times 7$ second-order covariance (beam matrix) $\Sigma$:

$$\Sigma = \langle (\vec{x} - \mu)(\vec{x} - \mu)^T \rangle$$

Propagation through a transfer matrix $R$ is computed analytically via similarity transformation:

$$\mu_{\text{out}} = R \mu_{\text{in}}, \qquad \Sigma_{\text{out}} = R \Sigma_{\text{in}} R^T$$

Because this computation depends only on the $7 \times 7$ matrix dimensions, tracking speed is independent of the real bunch particle count, allowing thousands of simulations per second.

### Instantiation

You can initialise a `ParameterBeam` from Courant-Snyder (Twiss) parameters:

```python
import torch
from cheetah import ParameterBeam

# Create from Twiss parameters
beam = ParameterBeam.from_twiss(
    beta_x=torch.tensor(3.14),
    beta_y=torch.tensor(4.2),
    alpha_x=torch.tensor(0.5),
    alpha_y=torch.tensor(-0.3),
    emittance_x=torch.tensor(1e-6),
    emittance_y=torch.tensor(1e-6),
    energy=torch.tensor(100e6),  # 100 MeV
)
```

Alternatively, you can provide the mean vector and covariance matrix directly:

```python
beam = ParameterBeam.from_parameters(
    mu=torch.zeros(7),
    cov=torch.eye(7) * 1e-6,
    energy=torch.tensor(100e6),
)
```

### Accessing Beam Properties

A `ParameterBeam` provides computed properties along horizontal and vertical axes:

```python
print("sigma_x:", beam.sigma_x)  # Horizontal RMS beam size (sqrt of cov[0, 0])
print("sigma_y:", beam.sigma_y)  # Vertical RMS beam size (sqrt of cov[2, 2])
print("beta_x:", beam.beta_x)  # Horizontal beta function
print("emittance_x:", beam.emittance_x)  # Geometric emittance
```

---

## `ParticleBeam`

A `ParticleBeam` tracks an ensemble of $N$ individual macroparticles represented as a tensor of shape $(N, 7)$:

$$\mathbf{X} = \begin{bmatrix} x_1 & px_1 & y_1 & py_1 & \tau_1 & \delta_1 & 1 \\ \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\ x_N & px_N & y_N & py_N & \tau_N & \delta_N & 1 \end{bmatrix}$$

This representation captures arbitrary non-Gaussian phase space distributions, microbunching, and nonlinear distortions. It is also required when simulating collective effects like 3D space charge via `SpaceChargeKick`.

### Instantiation

You can generate a Gaussian particle distribution from Twiss parameters by specifying the particle count:

```python
from cheetah import ParticleBeam

# Sample 100,000 particles from Twiss parameters
beam = ParticleBeam.from_twiss(
    beta_x=torch.tensor(3.14),
    beta_y=torch.tensor(4.2),
    energy=torch.tensor(100e6),
    num_particles=100_000,
)
```

Or pass a precomputed $(N, 7)$ tensor directly:

```python
particles = torch.randn(10_000, 7)
particles[:, 6] = 1.0  # Set homogeneous affine coordinate to 1.0

beam = ParticleBeam.from_distribution(
    particles=particles,
    energy=torch.tensor(100e6),
)
```

You can also import real particle distributions from simulation codes like Astra, openPMD, and Ocelot (see [Importing from Other Codes](importing_from_other_codes.md)).

---

## Converting Between Representations

Cheetah provides direct conversion methods between `ParameterBeam` and `ParticleBeam`:

### 1. Sampling a `ParticleBeam` from a `ParameterBeam`

To expand a moment-based `ParameterBeam` into an ensemble of discrete particles (e.g. before tracking through a `SpaceChargeKick` or `Aperture`):

```python
particle_beam = parameter_beam.to_particle_beam(num_particles=50_000)
```

### 2. Reducing a `ParticleBeam` to a `ParameterBeam`

To compute the equivalent statistical centroids and covariance matrix from a particle ensemble:

```python
parameter_beam = particle_beam.to_parameter_beam()
```

---

## Modeling Particle Loss with `Aperture`

While `ParameterBeam` assumes lossless linear transport, a `ParticleBeam` can track physical collimation and boundary losses via [`Aperture`](../accelerator.rst):

```python
import torch
import cheetah

# Define an aperture (rectangular or elliptical)
collimator = cheetah.Aperture(
    x_max=torch.tensor(2e-6),  # Horizontal half-width in meters
    y_max=torch.tensor(2e-6),  # Vertical half-height in meters
    shape="elliptical",  # "rectangular" or "elliptical"
    is_active=True,
)

# Track particle beam through collimator
outgoing_beam = collimator.track(beam)

# Inspect survival metrics
print("Initial particles:", beam.num_particles)
print("Survived particles:", outgoing_beam.num_particles_survived.item())
print(
    "Transmission (%):",
    (outgoing_beam.num_particles_survived / beam.num_particles * 100.0).item(),
)
```

### Survival Probabilities and Downstream Moments

Cheetah models collimation using particle survival probabilities:

- **`beam.survival_probabilities`**: A 1D tensor tracking the survival weight (or binary mask) of each macroparticle.
- **`beam.num_particles_survived`**: Total number of surviving particles (`survival_probabilities.sum()`).
- **Automatic moment reweighting**: All statistical properties on the surviving beam (`mu_x`, `sigma_x`, emittances, and `total_charge`) are automatically weighted by `survival_probabilities`. Particles stopped by the aperture do not contribute to downstream envelope measurements.
- **ParameterBeam warning**: Passing a `ParameterBeam` through an active `Aperture` emits a `PhysicsWarning` because moment-based beams cannot model transverse clipping.
