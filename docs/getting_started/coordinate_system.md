# Coordinate System in Cheetah

In _Cheetah_, the coordinates of the particles are defined by a 7-dimensional vector

$$\vec{x} = (x, px, y, py, \tau, \delta, 1)$$

The first 6 values are the canonical variables in the phase space.
The trailing $1$ is augmented as in affine transformation, for convenient calculations of thin-lens kicks, misalignmenents, etc.

The phase space coordinates are defined in the curvelinear coordinate system with respect to a reference particle at a position $s$ along the beamline:

- $x$ is the horizontal position in m
- $y$ is the vertical position in m
- $px$ is the normalized horizontal momentum, dimensionless
- $py$ is the normalized vertical momentum, dimensionless
- $\tau$ is _approximately_ the longitudinal offset with respect to the reference particle in m
- $\delta$ is the energy offset over the reference momentum, dimensionless

The new variables are

$$ px = P_x / p_0  $$

$P_x$ is the horizontal momentum and $p_0$ is the _reference momentum_.

$$ \tau = c\Delta t = ct- \frac{s}{\beta_0} $$

$s$ is the independent variable, denoting the position of the reference particle.
$t$ is the time when the particle arrives at position $z$.
In this notation, bunch head (particles arriving earlier than the reference particle) would have $\tau<0$ as $t<t_0$. The reference particle will have $\tau=0$.

$$ \delta= \frac{E-E_0}{p_0 c} = \frac{E}{p_0 c} - \frac{1}{\beta_0} $$

Here, $E$ is the energy of a particle. $E_0$ is the energy of the reference particle, i.e. _reference energy_. $p_0$ is the reference momentum.

## Relation with coordinates in other simulation tools

### OCELOT

<https://github.com/ocelot-collab/ocelot>

The coordinates are identical as the ones used in OCELOT.

Note that in OCELOT the energy has the unit of GeV, while in Cheetah the energy is in eV.

### Bmad

<https://www.classe.cornell.edu/bmad/>

The transverse coordinates $(x, px, y, py)$ are identical to the Bmad coordinates.

The longitudinal coordinate in Bmad is defined as

$$ z_\text{(Bmad)} = -\beta c \Delta t $$

$$ \tau_\text{(Cheetah)} = -\frac{1}{\beta} z_\text{(Bmad)} $$

In Bmad, the sixth dimension (longitudinal momentum) is defined as the momentum offset over the reference momentum

$$ p_{z, \text{(Bmad)}} = \frac{p-p_0}{p_0}$$

### Mad-X

<https://madx.web.cern.ch/madx/>

The longidutinal coordinate has an opposite sign

$$ \tau_\text{(Cheetah)} = - z_\text{(MAD)} $$

The longitudinal momentum is identical

$$ \delta_\text{(Cheetah)} = p_{t,\text{(MAD)}} $$

## Conversion to trace space notation

In many literatures, the trace space, or _slope_ notation is used.

$$ x' := \frac{dx}{ds} $$

In paraxial approximation, they are approximately the same $px \sim x'$.

In general

$$ x' = \frac{p_x}{\sqrt{p^2 - p_x^2 - p_y^2}} (1+gx) $$

Note that $p_s := \sqrt{p^2 - p_x^2 - p_y^2}$ is the longitudinal momentum, $g = 1/\rho$ is the curvature of the trajectory.

## Particle Species

Beams in Cheetah have an associated `Species` that defines the charge and mass of the particles being simulated. This is used to calculate relativistic factors and track physics updates.

### Predefined Species

Cheetah provides several predefined species that can be resolved automatically by name:
- `"electron"`
- `"positron"`
- `"proton"`
- `"antiproton"`
- `"deuteron"`

When creating a beam, pass the species name to populate its mass and charge:

```python
import torch
import cheetah

beam = cheetah.ParameterBeam.from_twiss(
    beta_x=torch.tensor(3.14),
    beta_y=torch.tensor(3.14),
    energy=torch.tensor(1e7),
    species="electron"
)
```

### Custom Species

For custom particle species (such as heavy ions), you can define a custom `Species` object by specifying its name, charge, and mass:

```python
import torch
import cheetah

# Define a custom carbon ion species
carbon_ion = cheetah.Species(
    name="carbon_ion",
    num_elementary_charges=torch.tensor(6.0),
    mass_eV=torch.tensor(11.178e9)
)

beam = cheetah.ParameterBeam.from_twiss(
    beta_x=torch.tensor(3.14),
    beta_y=torch.tensor(3.14),
    energy=torch.tensor(1e7),
    species=carbon_ion
)
```
