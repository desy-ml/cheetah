# Coordinate System in Cheetah

In _Cheetah_, the coordinates of the particles are defined by a 7-dimensional vector

$$\vec{x} = (x, xp, y, yp, \tau, \delta, 1)$$

The first 6 values are the canonical variables in the phase space.
The trailing $1$ is augmented as in affine transformation, for convenient calculations of thin-length kicks, misalignmenents, etc.

The phase space coordinates are defined in the curvelinear coordinate system with respect to a reference particle at a position $s$ along the beamline:

- $x$ is the horizontal position in m
- $y$ is the vertical position in m
- $xp$ is the normalized horizontal momentum, dimensionless
- $yp$ is the normalized vertical momentum, dimensionless
- $\tau$ is _approximately_ the longitudinal offset with respect to the reference particle in m
- $\delta$ is the energy offset over the reference momentum, dimensionless

The new variables are

$$ xp = p_x / p_0  $$

$p_x$ is the horizontal momentum and $p_0$ is the _reference momentum_.

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

The transverse coordinates $(x, xp, y, yp)$ are identical to the Bmad coordinates.

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

In paraxial approximation, they are approximately the same $xp \sim x'$.

In general

$$ x' = \frac{p_x}{\sqrt{p^2 - p_x^2 - p_y^2}} (1+gx) $$

Note that $p_s := \sqrt{p^2 - p_x^2 - p_y^2}$ is the longitudinal momentum, $g = 1/\rho$ is the curvature of the trajectory.
