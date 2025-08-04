import itertools
from typing import Literal

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import constants
from scipy.ndimage import gaussian_filter
from torch.distributions import MultivariateNormal

from cheetah.particles.beam import Beam
from cheetah.particles.species import Species
from cheetah.utils import (
    elementwise_linspace,
    format_axis_with_prefixed_unit,
    unbiased_weighted_covariance,
    unbiased_weighted_covariance_matrix,
    unbiased_weighted_std,
    verify_device_and_dtype,
)


class ParticleBeam(Beam):
    """
    Beam of charged particles, where each particle is simulated.

    :param particles: List of 7-dimensional particle vectors.
    :param energy: Reference energy of the beam in eV.
    :param particle_charges: Charges of the macroparticles in the beam in C.
    :param survival_probabilities: Vector of probabilities that each particle has
        survived (i.e. not been lost), where 1.0 means the particle has survived and
        0.0 means the particle has been lost. Defaults to ones.
    :param s: Position along the beamline of the reference particle in meters.
    :param species: Particle species of the beam. Defaults to electron.
    :param device: Device to move the beam's particle array to. If set to `"auto"` a
        CUDA GPU is selected if available. The CPU is used otherwise.
    :param dtype: Data type of the generated particles.
    """

    PRETTY_DIMENSION_LABELS = {
        "x": r"$x$",
        "px": r"$p_x$",
        "y": r"$y$",
        "py": r"$p_y$",
        "tau": r"$\tau$",
        "p": r"$\delta$",
    }
    UNVECTORIZED_NUM_ATTR_DIMS = Beam.UNVECTORIZED_NUM_ATTR_DIMS | {
        "particles": 2,
        "particle_charges": 1,
        "survival_probabilities": 1,
        "x": 1,
        "px": 1,
        "y": 1,
        "py": 1,
        "tau": 1,
        "p": 1,
    }

    def __init__(
        self,
        particles: torch.Tensor,
        energy: torch.Tensor,
        particle_charges: torch.Tensor | None = None,
        survival_probabilities: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        species: Species | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        device, dtype = verify_device_and_dtype(
            [particles, energy, particle_charges, survival_probabilities, s],
            device,
            dtype,
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        assert (
            particles.shape[-2] > 0 and particles.shape[-1] == 7
        ), "Particle vectors must be 7-dimensional."

        self.species = (
            species.to(**factory_kwargs)
            if species is not None
            else Species("electron", **factory_kwargs)
        )

        self.register_buffer_or_parameter(
            "particles", torch.as_tensor(particles, **factory_kwargs)
        )
        self.register_buffer_or_parameter(
            "energy", torch.as_tensor(energy, **factory_kwargs)
        )
        self.register_buffer_or_parameter(
            "particle_charges",
            (
                torch.as_tensor(particle_charges, **factory_kwargs)
                if particle_charges is not None
                else torch.full(
                    (particles.shape[-2],),
                    self.species.charge_coulomb,
                    **factory_kwargs,
                )
            ),
        )
        self.register_buffer_or_parameter(
            "survival_probabilities",
            (
                torch.as_tensor(survival_probabilities, **factory_kwargs)
                if survival_probabilities is not None
                else torch.ones(particles.shape[-2], **factory_kwargs)
            ),
        )
        self.register_buffer_or_parameter(
            "s", torch.as_tensor(s if s is not None else 0.0, **factory_kwargs)
        )

    @classmethod
    def from_parameters(
        cls,
        num_particles: int = 100_000,
        mu_x: torch.Tensor | None = None,
        mu_px: torch.Tensor | None = None,
        mu_y: torch.Tensor | None = None,
        mu_py: torch.Tensor | None = None,
        mu_tau: torch.Tensor | None = None,
        mu_p: torch.Tensor | None = None,
        sigma_x: torch.Tensor | None = None,
        sigma_px: torch.Tensor | None = None,
        sigma_y: torch.Tensor | None = None,
        sigma_py: torch.Tensor | None = None,
        sigma_tau: torch.Tensor | None = None,
        sigma_p: torch.Tensor | None = None,
        cov_xpx: torch.Tensor | None = None,
        cov_ypy: torch.Tensor | None = None,
        cov_taup: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
        total_charge: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        species: Species | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ParticleBeam":
        """
        Generate Cheetah Beam of random particles.

        :param num_particles: Number of particles to generate.
        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_px: Center of the particle distribution on px, dimensionless.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_py: Center of the particle distribution on py , dimensionless.
        :param mu_tau: Center of the particle distribution on tau (longitudinal) in
            meters.
        :param mu_p: Center of the particle distribution on p, dimensionless.
        :param sigma_x: Sigma of the particle distribution in x direction in meters.
        :param sigma_px: Sigma of the particle distribution in px direction,
            dimensionless.
        :param sigma_y: Sigma of the particle distribution in y direction in meters.
        :param sigma_py: Sigma of the particle distribution in py direction,
            dimensionless.
        :param sigma_tau: Sigma of the particle distribution in longitudinal direction,
            in meters.
        :param sigma_p: Sigma of the particle distribution in longitudinal momentum,
            dimensionless.
        :param cov_xpx: Correlation between x and px.
        :param cov_ypy: Correlation between y and py.
        :param cov_taup: Correlation between s and p.
        :param energy: Energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        :param s: Position along the beamline of the reference particle in meters.
        :param species: Particle species of the beam. Defaults to electron.
        :param device: Device to move the beam's particle array to. If set to `"auto"` a
            CUDA GPU is selected if available. The CPU is used otherwise.
        :param dtype: Data type of the generated particles.
        :return: ParticleBeam with random particles.
        """
        # Extract device and dtype from given arguments
        device, dtype = verify_device_and_dtype(
            [
                mu_x,
                mu_px,
                mu_y,
                mu_py,
                mu_tau,
                mu_p,
                sigma_x,
                sigma_px,
                sigma_y,
                sigma_py,
                sigma_tau,
                sigma_p,
                cov_xpx,
                cov_ypy,
                cov_taup,
                energy,
                total_charge,
                s,
            ],
            device,
            dtype,
        )
        factory_kwargs = {"device": device, "dtype": dtype}

        # Set default values without function call in function signature
        mu_x = mu_x if mu_x is not None else torch.tensor(0.0, **factory_kwargs)
        mu_px = mu_px if mu_px is not None else torch.tensor(0.0, **factory_kwargs)
        mu_y = mu_y if mu_y is not None else torch.tensor(0.0, **factory_kwargs)
        mu_py = mu_py if mu_py is not None else torch.tensor(0.0, **factory_kwargs)
        mu_tau = mu_tau if mu_tau is not None else torch.tensor(0.0, **factory_kwargs)
        mu_p = mu_p if mu_p is not None else torch.tensor(0.0, **factory_kwargs)
        sigma_x = (
            sigma_x if sigma_x is not None else torch.tensor(175e-6, **factory_kwargs)
        )
        sigma_px = (
            sigma_px if sigma_px is not None else torch.tensor(4e-6, **factory_kwargs)
        )
        sigma_y = (
            sigma_y if sigma_y is not None else torch.tensor(175e-6, **factory_kwargs)
        )
        sigma_py = (
            sigma_py if sigma_py is not None else torch.tensor(4e-6, **factory_kwargs)
        )
        sigma_tau = (
            sigma_tau if sigma_tau is not None else torch.tensor(8e-6, **factory_kwargs)
        )
        sigma_p = (
            sigma_p if sigma_p is not None else torch.tensor(2e-3, **factory_kwargs)
        )
        cov_xpx = (
            cov_xpx if cov_xpx is not None else torch.tensor(0.0, **factory_kwargs)
        )
        cov_ypy = (
            cov_ypy if cov_ypy is not None else torch.tensor(0.0, **factory_kwargs)
        )
        cov_taup = (
            cov_taup if cov_taup is not None else torch.tensor(0.0, **factory_kwargs)
        )

        mu_x, mu_px, mu_y, mu_py, mu_tau, mu_p = torch.broadcast_tensors(
            mu_x, mu_px, mu_y, mu_py, mu_tau, mu_p
        )
        mean = torch.stack([mu_x, mu_px, mu_y, mu_py, mu_tau, mu_p], dim=-1)

        (
            sigma_x,
            sigma_px,
            cov_xpx,
            sigma_y,
            sigma_py,
            cov_ypy,
            sigma_tau,
            sigma_p,
            cov_taup,
        ) = torch.broadcast_tensors(
            sigma_x,
            sigma_px,
            cov_xpx,
            sigma_y,
            sigma_py,
            cov_ypy,
            sigma_tau,
            sigma_p,
            cov_taup,
        )
        cov = torch.zeros(*sigma_x.shape, 6, 6, **factory_kwargs)
        cov[..., 0, 0] = sigma_x**2
        cov[..., 0, 1] = cov_xpx
        cov[..., 1, 0] = cov_xpx
        cov[..., 1, 1] = sigma_px**2
        cov[..., 2, 2] = sigma_y**2
        cov[..., 2, 3] = cov_ypy
        cov[..., 3, 2] = cov_ypy
        cov[..., 3, 3] = sigma_py**2
        cov[..., 4, 4] = sigma_tau**2
        cov[..., 4, 5] = cov_taup
        cov[..., 5, 4] = cov_taup
        cov[..., 5, 5] = sigma_p**2

        return cls.from_distribution(
            mu=mean,
            cov=cov,
            num_particles=num_particles,
            energy=energy,
            total_charge=total_charge,
            s=s,
            species=species,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_distribution(
        cls,
        mu: torch.Tensor,
        cov: torch.Tensor,
        num_particles: int = 100_000,
        energy: torch.Tensor | None = None,
        total_charge: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        species: Species | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ParticleBeam":
        """
        Generate Cheetah Beam of random particles from a multivariate normal
        distribution.

        :param num_particles: Number of particles to generate.
        :param mu: Mean of the multivariate normal distribution.
        :param cov: Covariance matrix of the multivariate normal distribution.
        :param energy: Energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        :param s: Position along the beamline of the reference particle in meters.
        :param species: Particle species of the beam. Defaults to electron.
        :param device: Device to move the beam's particle array to. If set to `"auto"` a
            CUDA GPU is selected if available. The CPU is used otherwise.
        :param dtype: Data type of the generated particles.
        :return: ParticleBeam with random particles.
        """
        # Extract device and dtype from given arguments
        device, dtype = verify_device_and_dtype(
            [mu, cov, energy, total_charge, s], device, dtype
        )
        factory_kwargs = {"device": device, "dtype": dtype}

        species = (
            species.to(**factory_kwargs)
            if species is not None
            else Species("electron", **factory_kwargs)
        )

        # Set default values without function call in function signature
        energy = energy if energy is not None else torch.tensor(1e8, **factory_kwargs)
        total_charge = (
            total_charge
            if total_charge is not None
            else species.charge_coulomb * num_particles
        )
        particle_charges = (
            torch.ones((*total_charge.shape, num_particles), **factory_kwargs)
            * total_charge.unsqueeze(-1)
            / num_particles
        )

        vector_shape = torch.broadcast_shapes(mu.shape[:-1], cov.shape[:-2])
        mu = mu.expand(*vector_shape, 6)
        cov = cov.expand(*vector_shape, 6, 6)
        particles = torch.ones((*vector_shape, num_particles, 7), **factory_kwargs)
        distributions = [
            MultivariateNormal(sample_mu, covariance_matrix=sample_cov)
            for sample_mu, sample_cov in zip(mu.view(-1, 6), cov.view(-1, 6, 6))
        ]
        particles[..., :6] = torch.stack(
            [distribution.rsample((num_particles,)) for distribution in distributions],
            dim=0,
        ).view(*vector_shape, num_particles, 6)

        return cls(
            particles,
            energy,
            particle_charges=particle_charges,
            s=s,
            species=species,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_twiss(
        cls,
        num_particles: int = 100_000,
        beta_x: torch.Tensor | None = None,
        alpha_x: torch.Tensor | None = None,
        emittance_x: torch.Tensor | None = None,
        beta_y: torch.Tensor | None = None,
        alpha_y: torch.Tensor | None = None,
        emittance_y: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
        sigma_tau: torch.Tensor | None = None,
        sigma_p: torch.Tensor | None = None,
        cov_taup: torch.Tensor | None = None,
        total_charge: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        species: Species | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ParticleBeam":
        # Extract device and dtype from given arguments
        device, dtype = verify_device_and_dtype(
            [
                beta_x,
                alpha_x,
                emittance_x,
                beta_y,
                alpha_y,
                emittance_y,
                energy,
                sigma_tau,
                sigma_p,
                cov_taup,
                total_charge,
                s,
            ],
            device,
            dtype,
        )
        factory_kwargs = {"device": device, "dtype": dtype}

        # Set default values without function call in function signature
        beta_x = beta_x if beta_x is not None else torch.tensor(0.0, **factory_kwargs)
        alpha_x = (
            alpha_x if alpha_x is not None else torch.tensor(0.0, **factory_kwargs)
        )
        emittance_x = (
            emittance_x
            if emittance_x is not None
            else torch.tensor(7.1971891e-13, **factory_kwargs)
        )
        beta_y = beta_y if beta_y is not None else torch.tensor(0.0, **factory_kwargs)
        alpha_y = (
            alpha_y if alpha_y is not None else torch.tensor(0.0, **factory_kwargs)
        )
        emittance_y = (
            emittance_y
            if emittance_y is not None
            else torch.tensor(7.1971891e-13, **factory_kwargs)
        )
        energy = energy if energy is not None else torch.tensor(1e8, **factory_kwargs)
        sigma_tau = (
            sigma_tau if sigma_tau is not None else torch.tensor(1e-6, **factory_kwargs)
        )
        sigma_p = (
            sigma_p if sigma_p is not None else torch.tensor(1e-6, **factory_kwargs)
        )
        cov_taup = (
            cov_taup if cov_taup is not None else torch.tensor(0.0, **factory_kwargs)
        )

        sigma_x = torch.sqrt(beta_x * emittance_x)
        sigma_px = torch.sqrt(emittance_x * (1 + alpha_x**2) / beta_x)
        sigma_y = torch.sqrt(beta_y * emittance_y)
        sigma_py = torch.sqrt(emittance_y * (1 + alpha_y**2) / beta_y)
        cov_xpx = -emittance_x * alpha_x
        cov_ypy = -emittance_y * alpha_y

        return cls.from_parameters(
            num_particles=num_particles,
            mu_x=torch.tensor(0.0, **factory_kwargs),
            mu_px=torch.tensor(0.0, **factory_kwargs),
            mu_y=torch.tensor(0.0, **factory_kwargs),
            mu_py=torch.tensor(0.0, **factory_kwargs),
            sigma_x=sigma_x,
            sigma_px=sigma_px,
            sigma_y=sigma_y,
            sigma_py=sigma_py,
            sigma_tau=sigma_tau,
            sigma_p=sigma_p,
            energy=energy,
            cov_taup=cov_taup,
            cov_xpx=cov_xpx,
            cov_ypy=cov_ypy,
            total_charge=total_charge,
            s=s,
            species=species,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def uniform_3d_ellipsoid(
        cls,
        num_particles: int = 100_000,
        radius_x: torch.Tensor | None = None,
        radius_y: torch.Tensor | None = None,
        radius_tau: torch.Tensor | None = None,
        sigma_px: torch.Tensor | None = None,
        sigma_py: torch.Tensor | None = None,
        sigma_p: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
        total_charge: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        species: Species | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Generate a particle beam with spatially uniformly distributed particles inside
        an ellipsoid, i.e. a waterbag distribution.

        Note that:
         - The generated particles do not have correlation in the momentum directions,
           and by default a cold beam with no divergence is generated.

        :param num_particles: Number of particles to generate.
        :param radius_x: Radius of the ellipsoid in x direction in meters.
        :param radius_y: Radius of the ellipsoid in y direction in meters.
        :param radius_tau: Radius of the ellipsoid in tau (longitudinal) direction
            in meters.
        :param sigma_px: Sigma of the particle distribution in px direction,
            dimensionless, default is 0.
        :param sigma_py: Sigma of the particle distribution in py direction,
            dimensionless, default is 0.
        :param sigma_p: Sigma of the particle distribution in p, dimensionless.
        :param energy: Reference energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        :param s: Position along the beamline of the reference particle in meters.
        :param species: Particle species of the beam. Defaults to electron.
        :param device: Device to move the beam's particle array to. If set to `"auto"` a
            CUDA GPU is selected if available. The CPU is used otherwise.
        :param dtype: Data type of the generated particles.
        :return: ParticleBeam with uniformly distributed particles inside an ellipsoid.
        """
        # Extract device and dtype from given arguments
        device, dtype = verify_device_and_dtype(
            [
                radius_x,
                radius_y,
                radius_tau,
                sigma_px,
                sigma_py,
                sigma_p,
                energy,
                total_charge,
                s,
            ],
            device,
            dtype,
        )
        factory_kwargs = {"device": device, "dtype": dtype}

        # Set default values without function call in function signature
        # NOTE that this does not need to be done for values that are passed to the
        # Gaussian beam generation.
        radius_x = (
            radius_x if radius_x is not None else torch.tensor(1e-3, **factory_kwargs)
        )
        radius_y = (
            radius_y if radius_y is not None else torch.tensor(1e-3, **factory_kwargs)
        )
        radius_tau = (
            radius_tau
            if radius_tau is not None
            else torch.tensor(1e-3, **factory_kwargs)
        )

        # Generate an uncorrelated Gaussian beam
        beam = cls.from_parameters(
            num_particles=num_particles,
            mu_px=torch.tensor(0.0, **factory_kwargs),
            mu_py=torch.tensor(0.0, **factory_kwargs),
            sigma_x=radius_x,  # Only a placeholder, will be overwritten
            sigma_px=sigma_px,
            sigma_y=radius_y,  # Only a placeholder, will be overwritten
            sigma_py=sigma_py,
            sigma_tau=radius_tau,  # Only a placeholder, will be overwritten
            sigma_p=sigma_p,
            energy=energy,
            total_charge=total_charge,
            s=s,
            species=species,
            device=device,
            dtype=dtype,
        )

        # Extract the vector dimension of the beam
        vector_shape = beam.sigma_x.shape

        # Generate random particles in unit sphere in polar coodinates
        # r: radius, 3rd root for uniform distribution in sphere volume
        # theta: polar angle, arccos for uniform distribution in sphere surface
        # phi: azimuthal angle, uniform between 0 and 2*pi
        r = torch.pow(torch.rand(*vector_shape, num_particles, **factory_kwargs), 1 / 3)
        theta = torch.arccos(
            2 * torch.rand(*vector_shape, num_particles, **factory_kwargs) - 1
        )
        phi = torch.rand(*vector_shape, num_particles, **factory_kwargs) * 2 * torch.pi

        # Convert to Cartesian coordinates
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        tau = r * torch.cos(theta)

        # Replace the spatial coordinates with the generated ones.
        # This involves distorting the unit sphere into the desired ellipsoid.
        beam.x = x * radius_x.unsqueeze(-1)
        beam.y = y * radius_y.unsqueeze(-1)
        beam.tau = tau * radius_tau.unsqueeze(-1)

        return beam

    @classmethod
    def make_linspaced(
        cls,
        num_particles: int = 10,
        mu_x: torch.Tensor | None = None,
        mu_px: torch.Tensor | None = None,
        mu_y: torch.Tensor | None = None,
        mu_py: torch.Tensor | None = None,
        mu_tau: torch.Tensor | None = None,
        mu_p: torch.Tensor | None = None,
        sigma_x: torch.Tensor | None = None,
        sigma_px: torch.Tensor | None = None,
        sigma_y: torch.Tensor | None = None,
        sigma_py: torch.Tensor | None = None,
        sigma_tau: torch.Tensor | None = None,
        sigma_p: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
        total_charge: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        species: Species | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ParticleBeam":
        """
        Generate Cheetah Beam of *n* linspaced particles.

        :param n: Number of particles to generate.
        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_px: Center of the particle distribution on px, dimensionless.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_py: Center of the particle distribution on py , dimensionless.
        :param mu_tau: Center of the particle distribution on tau (longitudinal) in
            meters.
        :param mu_p: Center of the particle distribution on p, dimensionless.
        :param sigma_x: Sigma of the particle distribution in x direction in meters.
        :param sigma_px: Sigma of the particle distribution in px direction,
            dimensionless.
        :param sigma_y: Sigma of the particle distribution in y direction in meters.
        :param sigma_py: Sigma of the particle distribution in py direction,
            dimensionless.
        :param sigma_tau: Sigma of the particle distribution in longitudinal direction,
            in meters.
        :param sigma_p: Sigma of the particle distribution in p, dimensionless.
        :param energy: Energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        :param s: Position along the beamline of the reference particle in meters.
        :param species: Particle species of the beam. Defaults to electron.
        :param device: Device to move the beam's particle array to. If set to `"auto"` a
            CUDA GPU is selected if available. The CPU is used otherwise.
        :param dtype: Data type of the generated particles.
        :return: ParticleBeam with *n* linspaced particles.
        """
        # Extract device and dtype from given arguments
        device, dtype = verify_device_and_dtype(
            [
                mu_x,
                mu_px,
                mu_y,
                mu_py,
                sigma_x,
                sigma_px,
                sigma_y,
                sigma_py,
                sigma_tau,
                sigma_p,
                energy,
                total_charge,
                s,
            ],
            device,
            dtype,
        )
        factory_kwargs = {"device": device, "dtype": dtype}

        species = species if species is not None else Species("electron")

        # Set default values without function call in function signature
        mu_x = mu_x if mu_x is not None else torch.tensor(0.0, **factory_kwargs)
        mu_px = mu_px if mu_px is not None else torch.tensor(0.0, **factory_kwargs)
        mu_y = mu_y if mu_y is not None else torch.tensor(0.0, **factory_kwargs)
        mu_py = mu_py if mu_py is not None else torch.tensor(0.0, **factory_kwargs)
        mu_tau = mu_tau if mu_tau is not None else torch.tensor(0.0, **factory_kwargs)
        mu_p = mu_p if mu_p is not None else torch.tensor(0.0, **factory_kwargs)
        sigma_x = (
            sigma_x if sigma_x is not None else torch.tensor(175e-9, **factory_kwargs)
        )
        sigma_px = (
            sigma_px if sigma_px is not None else torch.tensor(2e-7, **factory_kwargs)
        )
        sigma_y = (
            sigma_y if sigma_y is not None else torch.tensor(175e-9, **factory_kwargs)
        )
        sigma_py = (
            sigma_py if sigma_py is not None else torch.tensor(2e-7, **factory_kwargs)
        )
        sigma_tau = (
            sigma_tau if sigma_tau is not None else torch.tensor(1e-6, **factory_kwargs)
        )
        sigma_p = (
            sigma_p if sigma_p is not None else torch.tensor(1e-6, **factory_kwargs)
        )
        energy = energy if energy is not None else torch.tensor(1e8, **factory_kwargs)
        total_charge = (
            total_charge
            if total_charge is not None
            else species.charge_coulomb * num_particles
        )
        particle_charges = (
            torch.ones((*total_charge.shape, num_particles), **factory_kwargs)
            * total_charge.unsqueeze(-1)
            / num_particles
        )

        vector_shape = torch.broadcast_shapes(
            mu_x.shape,
            mu_px.shape,
            mu_y.shape,
            mu_py.shape,
            mu_tau.shape,
            mu_p.shape,
            sigma_x.shape,
            sigma_px.shape,
            sigma_y.shape,
            sigma_py.shape,
            sigma_tau.shape,
            sigma_p.shape,
        )
        particles = torch.ones((*vector_shape, num_particles, 7), **factory_kwargs)

        particles[..., 0] = elementwise_linspace(
            mu_x - sigma_x, mu_x + sigma_x, num_particles
        )
        particles[..., 1] = elementwise_linspace(
            mu_px - sigma_px, mu_px + sigma_px, num_particles
        )
        particles[..., 2] = elementwise_linspace(
            mu_y - sigma_y, mu_y + sigma_y, num_particles
        )
        particles[..., 3] = elementwise_linspace(
            mu_py - sigma_py, mu_py + sigma_py, num_particles
        )
        particles[..., 4] = elementwise_linspace(
            mu_tau - sigma_tau, mu_tau + sigma_tau, num_particles
        )
        particles[..., 5] = elementwise_linspace(
            mu_p - sigma_p, mu_p + sigma_p, num_particles
        )

        return cls(
            particles=particles,
            energy=energy,
            particle_charges=particle_charges,
            s=s,
            species=species,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_ocelot(
        cls, parray, device: torch.device = None, dtype: torch.dtype = None
    ) -> "ParticleBeam":
        """Convert an Ocelot ParticleArray `parray` to a Cheetah Beam."""
        num_particles = parray.rparticles.shape[1]
        particles = torch.ones((num_particles, 7), device=device, dtype=dtype)
        particles[:, :6] = torch.as_tensor(
            parray.rparticles.transpose(), device=device, dtype=dtype
        )
        particle_charges = torch.as_tensor(parray.q_array, device=device, dtype=dtype)

        return cls(
            particles=particles,
            energy=1e9 * torch.as_tensor(parray.E),
            particle_charges=particle_charges,
            species=Species("electron"),
            device=device or torch.get_default_device(),
            dtype=dtype or torch.get_default_dtype(),
        )

    @classmethod
    def from_astra(
        cls, path: str, device: torch.device = None, dtype: torch.dtype = None
    ) -> "ParticleBeam":
        """Load an Astra particle distribution as a Cheetah Beam."""
        from cheetah.converters.astra import from_astrabeam

        particles, energy, particle_charges = from_astrabeam(path)

        particles_7d = torch.ones((particles.shape[0], 7), device=device, dtype=dtype)
        particles_7d[:, :6] = torch.as_tensor(particles, device=device, dtype=dtype)

        energy = torch.as_tensor(energy, device=device, dtype=dtype)
        particle_charges = torch.as_tensor(particle_charges, device=device, dtype=dtype)

        return cls(
            particles=particles_7d,
            energy=energy,
            particle_charges=particle_charges,
            species=Species("electron"),
            device=device or torch.get_default_device(),
            dtype=dtype or torch.get_default_dtype(),
        )

    @classmethod
    def from_openpmd_file(
        cls,
        path: str,
        energy: torch.Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ParticleBeam":
        """Load an openPMD particle group HDF5 file as a Cheetah `ParticleBeam`."""
        try:
            import pmd_beamphysics as openpmd
        except ImportError:
            raise ImportError(
                """To use the openPMD beam import, openPMD-beamphysics must be
                installed."""
            )

        particle_group = openpmd.ParticleGroup(path)
        return cls.from_openpmd_particlegroup(
            particle_group, energy, device=device, dtype=dtype
        )

    @classmethod
    def from_openpmd_particlegroup(
        cls,
        particle_group: "openpmd.ParticleGroup",  # noqa: F821
        energy: torch.Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ParticleBeam":
        """
        Create a Cheetah `ParticleBeam` from an openPMD `ParticleGroup` object.

        :param particle_group: openPMD `ParticleGroup` object.
        :param energy: Reference energy of the beam in eV.
        :param device: Device to move the beam's particle array to. If set to `"auto"` a
            CUDA GPU is selected if available. The CPU is used otherwise.
        :param dtype: Data type of the generated particles.
        """
        device, dtype = verify_device_and_dtype([energy], device, dtype)
        factory_kwargs = {"device": device, "dtype": dtype}

        species = Species(particle_group.species, **factory_kwargs)
        p0c = torch.sqrt(energy**2 - species.mass_eV**2)

        x = torch.as_tensor(particle_group.x, **factory_kwargs)
        y = torch.as_tensor(particle_group.y, **factory_kwargs)
        px = torch.as_tensor(particle_group.px, **factory_kwargs) / p0c
        py = torch.as_tensor(particle_group.py, **factory_kwargs) / p0c
        tau = (
            torch.as_tensor(particle_group.t, **factory_kwargs)
            * constants.speed_of_light
        )
        delta = (
            torch.as_tensor(particle_group.energy, **factory_kwargs) - energy
        ) / p0c

        particles = torch.stack([x, px, y, py, tau, delta, torch.ones_like(x)], dim=-1)
        particle_charges = torch.as_tensor(particle_group.weight, **factory_kwargs)
        survival_probabilities = torch.as_tensor(
            particle_group.status, **factory_kwargs
        )

        return cls(
            particles=particles,
            energy=energy,
            particle_charges=particle_charges,
            survival_probabilities=survival_probabilities,
            species=species,
            device=device,
            dtype=dtype,
        )

    def save_as_openpmd_h5(self, path: str) -> None:
        """
        Save the `ParticleBeam` as an openPMD particle group HDF5 file.

        :param path: Path to the file where the beam should be saved.
        """
        particle_group = self.to_openpmd_particlegroup()
        particle_group.write(path)

    def to_openpmd_particlegroup(self) -> "openpmd.ParticleGroup":  # noqa: F821
        """
        Convert the `ParticleBeam` to an openPMD `ParticleGroup` object.

        NOTE: openPMD uses boolean particle status flags, i.e. alive or dead. Cheetah's
            survival probabilities are converted to status flags by thresholding at 0.5.

        NOTE: At the moment this method only supports non-vectorised particle
            distributions.

        :return: openPMD `ParticleGroup` object with the `ParticleBeam`'s particles.
        """
        try:
            import pmd_beamphysics as openpmd
        except ImportError:
            raise ImportError(
                """To use the openPMD beam export, openPMD-beamphysics must be
                installed."""
            )

        # For now only support non-vectorised particle distributions
        if len(self.particles.shape) != 2:
            raise ValueError(
                "Only non-vectorised particle distributions are supported."
            )

        px = self.px * self.p0c
        py = self.py * self.p0c
        p_total = torch.sqrt(self.energies**2 - self.species.mass_eV**2)
        pz = torch.sqrt(p_total**2 - px**2 - py**2)
        t = self.tau / constants.speed_of_light
        # TODO: To be discussed
        status = self.survival_probabilities > 0.5

        data = {
            "x": self.x.numpy(),
            "y": self.y.numpy(),
            "z": self.tau.numpy(),
            "px": px.numpy(),
            "py": py.numpy(),
            "pz": pz.numpy(),
            "t": t.numpy(),
            "weight": self.particle_charges.numpy(),
            "status": status.numpy(),
            "species": self.species.name,
        }
        particle_group = openpmd.ParticleGroup(data=data)

        return particle_group

    def transformed_to(
        self,
        mu_x: torch.Tensor | None = None,
        mu_px: torch.Tensor | None = None,
        mu_y: torch.Tensor | None = None,
        mu_py: torch.Tensor | None = None,
        mu_tau: torch.Tensor | None = None,
        mu_p: torch.Tensor | None = None,
        sigma_x: torch.Tensor | None = None,
        sigma_px: torch.Tensor | None = None,
        sigma_y: torch.Tensor | None = None,
        sigma_py: torch.Tensor | None = None,
        sigma_tau: torch.Tensor | None = None,
        sigma_p: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
        total_charge: torch.Tensor | None = None,
        species: Species | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ParticleBeam":
        """
        Create version of this beam that is transformed to new beam parameters.

        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_px: Center of the particle distribution on px, dimensionless.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_py: Center of the particle distribution on py , dimensionless.
        :param mu_tau: Center of the particle distribution on tau (longitudinal) in
            meters.
        :param mu_p: Center of the particle distribution on p, dimensionless.
        :param sigma_x: Sigma of the particle distribution in x direction in meters.
        :param sigma_px: Sigma of the particle distribution in px direction,
            dimensionless.
        :param sigma_y: Sigma of the particle distribution in y direction in meters.
        :param sigma_py: Sigma of the particle distribution in py direction,
            dimensionless.
        :param sigma_tau: Sigma of the particle distribution in longitudinal direction,
            in meters.
        :param sigma_p: Sigma of the particle distribution in p, dimensionless.
        :param energy: Reference energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        :param species: Species of the particles in the beam.
        :param device: Device to move the beam's particle array to. If set to `"auto"` a
            CUDA GPU is selected if available. The CPU is used otherwise.
        :param dtype: Data type of the transformed particles.
        """
        device = device if device is not None else self.mu_x.device
        dtype = dtype if dtype is not None else self.mu_x.dtype

        mu_x = mu_x if mu_x is not None else self.mu_x
        mu_px = mu_px if mu_px is not None else self.mu_px
        mu_y = mu_y if mu_y is not None else self.mu_y
        mu_py = mu_py if mu_py is not None else self.mu_py
        mu_tau = mu_tau if mu_tau is not None else self.mu_tau
        mu_p = mu_p if mu_p is not None else self.mu_p
        sigma_x = sigma_x if sigma_x is not None else self.sigma_x
        sigma_px = sigma_px if sigma_px is not None else self.sigma_px
        sigma_y = sigma_y if sigma_y is not None else self.sigma_y
        sigma_py = sigma_py if sigma_py is not None else self.sigma_py
        sigma_tau = sigma_tau if sigma_tau is not None else self.sigma_tau
        sigma_p = sigma_p if sigma_p is not None else self.sigma_p
        energy = energy if energy is not None else self.energy
        if total_charge is None:
            particle_charges = self.particle_charges
        elif self.total_charge is None:  # Scale to the new charge
            total_charge = total_charge.to(
                device=self.particle_charges.device, dtype=self.particle_charges.dtype
            )
            particle_charges = self.particle_charges * total_charge / self.total_charge
        else:
            particle_charges = (
                torch.ones_like(self.particle_charges, device=device, dtype=dtype)
                * total_charge.unsqueeze(-1)
                / self.particle_charges.shape[-1]
            )
        species = species if species is not None else self.species

        mu_x, mu_px, mu_y, mu_py, mu_tau, mu_p = torch.broadcast_tensors(
            mu_x, mu_px, mu_y, mu_py, mu_tau, mu_p
        )
        new_mu = torch.stack([mu_x, mu_px, mu_y, mu_py, mu_tau, mu_p], dim=-1)
        sigma_x, sigma_px, sigma_y, sigma_py, sigma_tau, sigma_p = (
            torch.broadcast_tensors(
                sigma_x, sigma_px, sigma_y, sigma_py, sigma_tau, sigma_p
            )
        )
        new_sigma = torch.stack(
            [sigma_x, sigma_px, sigma_y, sigma_py, sigma_tau, sigma_p], dim=-1
        )

        old_mu = torch.stack(
            [self.mu_x, self.mu_px, self.mu_y, self.mu_py, self.mu_tau, self.mu_p],
            dim=-1,
        )
        old_sigma = torch.stack(
            [
                self.sigma_x,
                self.sigma_px,
                self.sigma_y,
                self.sigma_py,
                self.sigma_tau,
                self.sigma_p,
            ],
            dim=-1,
        )

        phase_space = self.particles[..., :6]
        phase_space = (
            (phase_space.transpose(-2, -1) - old_mu.unsqueeze(-1))
            / old_sigma.unsqueeze(-1)
            * new_sigma.unsqueeze(-1)
            + new_mu.unsqueeze(-1)
        ).transpose(-2, -1)

        particles = torch.ones(*phase_space.shape[:-1], 7)
        particles[..., :6] = phase_space

        return self.__class__(
            particles=particles,
            energy=energy,
            particle_charges=particle_charges,
            survival_probabilities=self.survival_probabilities,
            s=self.s,
            species=species,
            device=device,
            dtype=dtype,
        )

    def as_parameter_beam(self) -> "ParameterBeam":  # noqa: F821
        """
        Convert the the beam to a `ParameterBeam`.

        :return: `ParameterBeam` having the same parameters as this beam.
        """
        from cheetah.particles.parameter_beam import ParameterBeam  # No circular import

        return ParameterBeam(
            mu=(self.particles * self.survival_probabilities.unsqueeze(-1)).sum(dim=-2)
            / self.survival_probabilities.sum(dim=-1, keepdim=True),
            cov=unbiased_weighted_covariance_matrix(
                self.particles, weights=self.survival_probabilities
            ),
            energy=self.energy,
            total_charge=self.total_charge,
            device=self.particles.device,
            dtype=self.particles.dtype,
        )

    def linspaced(self, num_particles: int) -> "ParticleBeam":
        """
        Create a new beam with the same parameters as this beam, but with
        `num_particles` particles evenly distributed in the beam.

        :param num_particles: Number of particles to create.
        :return: New beam with `num_particles` particles.
        """
        return self.make_linspaced(
            num_particles=num_particles,
            mu_x=self.mu_x,
            mu_px=self.mu_px,
            mu_y=self.mu_y,
            mu_py=self.mu_py,
            mu_tau=self.mu_tau,
            mu_p=self.mu_p,
            sigma_x=self.sigma_x,
            sigma_px=self.sigma_px,
            sigma_y=self.sigma_y,
            sigma_py=self.sigma_py,
            sigma_tau=self.sigma_tau,
            sigma_p=self.sigma_p,
            energy=self.energy,
            total_charge=self.total_charge,
            particle_charges=self.particle_charges,
            survival_probabilities=self.survival_probabilities,
            s=self.s,
            species=self.species,
            device=self.particles.device,
            dtype=self.particles.dtype,
        )

    def randomly_subsampled(
        self,
        num_particles: int,
        adjust_particle_charges: bool = True,
        random_state: torch.Generator | None = None,
    ) -> "ParticleBeam":
        """
        Create a new beam with the same parameters as this beam, but with
        `num_particles` particles randomly sampled from the original beam.

        :param num_particles: Number of particles to sample.
        :param adjust_particle_charges: If True, the particle charges are adjusted
            to match the total charge of the old beam.
        :param random_state: Random state to use for thinning. If None, a new random
            state is created.
        :return: New beam with `num_particles` particles.
        """
        assert num_particles <= self.num_particles, (
            "Number of particles to sample must be less than or equal to the number of "
            "particles in the original beam."
        )

        if random_state is None:
            random_state = torch.Generator(device=self.particles.device)

        randomly_permuted_particle_indices = torch.randperm(
            self.num_particles, generator=random_state, device=random_state.device
        )
        subsampled_particle_indices = randomly_permuted_particle_indices[:num_particles]

        subsampled_particles = self.particles[subsampled_particle_indices]
        subsampled_particle_charges = self.particle_charges[subsampled_particle_indices]
        subsampled_survival_probabilities = self.survival_probabilities[
            subsampled_particle_indices
        ]

        subsampled_beam = self.__class__(
            particles=subsampled_particles,
            energy=self.energy,
            particle_charges=subsampled_particle_charges,
            survival_probabilities=subsampled_survival_probabilities,
            species=self.species,
            device=self.particles.device,
            dtype=self.particles.dtype,
        )

        if adjust_particle_charges:
            subsampled_beam.particle_charges *= (
                self.total_charge / subsampled_beam.total_charge
            )

        return subsampled_beam

    @classmethod
    def from_xyz_pxpypz(
        cls,
        xp_coordinates: torch.Tensor,
        energy: torch.Tensor,
        particle_charges: torch.Tensor | None = None,
        survival_probabilities: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        species: Species | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        Create a beam from a tensor of position and momentum coordinates in SI units.
        This tensor should have shape (..., n_particles, 7), where the last dimension
        is the moment vector $(x, p_x, y, p_y, z, p_z, 1)$.
        """
        beam = cls(
            particles=xp_coordinates.clone(),
            energy=energy,
            particle_charges=particle_charges,
            survival_probabilities=survival_probabilities,
            s=s,
            species=species,
            device=device,
            dtype=dtype,
        )

        p0 = (
            beam.relativistic_gamma
            * beam.relativistic_beta
            * beam.species.mass_kg
            * constants.speed_of_light
        )
        p = torch.sqrt(
            xp_coordinates[..., 1] ** 2
            + xp_coordinates[..., 3] ** 2
            + xp_coordinates[..., 5] ** 2
        )
        gamma = torch.sqrt(
            1 + (p / (beam.species.mass_kg * constants.speed_of_light)) ** 2
        )

        beam.particles[..., 1] = xp_coordinates[..., 1] / p0.unsqueeze(-1)
        beam.particles[..., 3] = xp_coordinates[..., 3] / p0.unsqueeze(-1)
        beam.particles[..., 4] = -xp_coordinates[
            ..., 4
        ] / beam.relativistic_beta.unsqueeze(-1)
        beam.particles[..., 5] = (gamma - beam.relativistic_gamma.unsqueeze(-1)) / (
            (beam.relativistic_beta * beam.relativistic_gamma).unsqueeze(-1)
        )

        return beam

    def to_xyz_pxpypz(self) -> torch.Tensor:
        """
        Extracts the position and momentum coordinates in SI units, from the
        beam's `particles`, and returns it as a tensor with shape (..., n_particles, 7).
        For each particle, the obtained vector is $(x, p_x, y, p_y, z, p_z, 1)$.
        """
        p0 = (
            self.relativistic_gamma
            * self.relativistic_beta
            * self.species.mass_kg
            * constants.speed_of_light
        )  # Reference momentum in (kg m/s)
        gamma = self.relativistic_gamma.unsqueeze(-1) * (
            1.0 + self.particles[..., 5] * self.relativistic_beta.unsqueeze(-1)
        )
        beta = torch.sqrt(1 - 1 / gamma**2)
        momentum = gamma * self.species.mass_kg * beta * constants.speed_of_light

        px = self.particles[..., 1] * p0.unsqueeze(-1)
        py = self.particles[..., 3] * p0.unsqueeze(-1)
        zs = self.particles[..., 4] * -self.relativistic_beta.unsqueeze(-1)
        p = torch.sqrt(momentum**2 - px**2 - py**2)

        xp_coords = self.particles.clone()
        xp_coords[..., 1] = px
        xp_coords[..., 3] = py
        xp_coords[..., 4] = zs
        xp_coords[..., 5] = p

        return xp_coords

    def plot_1d_distribution(
        self,
        dimension: Literal["x", "px", "y", "py", "tau", "p"],
        bins: int = 100,
        bin_range: tuple[float] | None = None,
        smoothing: float = 0.0,
        plot_kws: dict | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """
        Plot a 1D histogram of the given dimension of the particle distribution.

        :param dimension: Name of the dimension to plot. Should be one of
            `('x', 'px', 'y', 'py', 'tau', 'p')`.
        :param bins: Number of bins to use for the histogram.
        :param bin_range: Range of the bins to use for the histogram.
        :param smoothing: Standard deviation of the Gaussian kernel used to smooth the
            histogram.
        :param plot_kws: Additional keyword arguments to be passed to `plot` function of
            matplotlib used to plot the histogram data.
        :param ax: Matplotlib axes object to use for plotting.
        :return: Matplotlib axes object with the plot.
        """
        if ax is None:
            _, ax = plt.subplots()

        x_array = getattr(self, dimension).cpu().detach().numpy()
        histogram, edges = np.histogram(x_array, bins=bins, range=bin_range)
        centers = (edges[:-1] + edges[1:]) / 2

        if smoothing:
            histogram = gaussian_filter(histogram, smoothing)

        ax.plot(
            centers,
            histogram / histogram.max(),
            **{"color": "black"} | (plot_kws or {}),
        )
        ax.set_xlabel(f"{self.PRETTY_DIMENSION_LABELS[dimension]}")

        # Handle units
        if dimension in ("x", "y", "tau"):
            base_unit = "m"

        if dimension in ("x", "y", "tau"):
            format_axis_with_prefixed_unit(ax.xaxis, base_unit, centers)

        return ax

    def plot_2d_distribution(
        self,
        x_dimension: Literal["x", "px", "y", "py", "tau", "p"],
        y_dimension: Literal["x", "px", "y", "py", "tau", "p"],
        style: Literal["histogram", "contour"] = "histogram",
        bins: int = 100,
        bin_ranges: tuple[tuple[float]] | None = None,
        histogram_smoothing: float = 0.0,
        contour_smoothing: float = 3.0,
        pcolormesh_kws: dict | None = None,
        contour_kws: dict | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """
        Plot a 2D histogram of the given dimensions of the particle distribution.

        :param x_dimension: Name of the x dimension to plot. Should be one of
            `('x', 'px', 'y', 'py', 'tau', 'p')`.
        :param y_dimension: Name of the y dimension to plot. Should be one of
            `('x', 'px', 'y', 'py', 'tau', 'p')`.
        :param style: Style of the plot. Should be one of `('histogram', 'contour')`.
        :param bins: Number of bins to use for the histogram in both dimensions.
        :param bin_ranges: Ranges of the bins to use for the histogram in each
            dimension.
        :param smoothing: Standard deviation of the Gaussian kernel used to smooth the
            histogram.
        :param pcolormesh_kws: Additional keyword arguments to be passed to `pcolormesh`
            function of matplotlib used to plot the histogram data.
        :param contour_kws: Additional keyword arguments to be passed to `contour`
            function of matplotlib used to plot the histogram data.
        :param ax: Matplotlib axes object to use for plotting.
        :return: Matplotlib axes object with the plot.
        """
        if ax is None:
            _, ax = plt.subplots()

        histogram, x_edges, y_edges = np.histogram2d(
            getattr(self, x_dimension).cpu().detach().numpy(),
            getattr(self, y_dimension).cpu().detach().numpy(),
            bins=bins,
            range=bin_ranges,
        )
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        # Post-process and plot
        smoothed_histogram = gaussian_filter(histogram, histogram_smoothing)
        clipped_histogram = np.where(smoothed_histogram > 1, smoothed_histogram, np.nan)
        if style == "histogram":
            ax.pcolormesh(
                x_edges,
                y_edges,
                clipped_histogram.T / smoothed_histogram.max(),
                **{"cmap": "rainbow"} | (pcolormesh_kws or {}),
            )
        elif style == "contour":
            contour_histogram = gaussian_filter(histogram, contour_smoothing)

            ax.contour(
                x_centers,
                y_centers,
                contour_histogram.T / contour_histogram.max(),
                **{"levels": 3} | (contour_kws or {}),
            )

        ax.set_xlabel(f"{self.PRETTY_DIMENSION_LABELS[x_dimension]}")
        ax.set_ylabel(f"{self.PRETTY_DIMENSION_LABELS[y_dimension]}")

        # Handle units
        if x_dimension in ("x", "y", "tau"):
            x_base_unit = "m"

        if y_dimension in ("x", "y", "tau"):
            y_base_unit = "m"

        if x_dimension in ("x", "y", "tau"):
            format_axis_with_prefixed_unit(ax.xaxis, x_base_unit, x_centers)

        if y_dimension in ("x", "y", "tau"):
            format_axis_with_prefixed_unit(ax.yaxis, y_base_unit, y_centers)

        return ax

    def plot_distribution(
        self,
        dimensions: tuple[str, ...] = ("x", "px", "y", "py", "tau", "p"),
        bins: int = 100,
        bin_ranges: Literal["same"] | tuple[float] | list[tuple[float]] | None = None,
        plot_1d_kws: dict | None = None,
        plot_2d_kws: dict | None = None,
        axs: list[plt.Axes] | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Plot of coordinates projected into 2D planes.

        :param dimensions: Tuple of dimensions to plot. Should be a subset of
            `('x', 'px', 'y', 'py', 'tau', 'p')`.
        :param contour: If `True`, overlay contour lines on the 2D histogram plots.
        :param bins: Number of bins to use for the histograms.
        :param bin_ranges: Ranges of the bins to use for the histograms. If set to
            `"unit_same"`, the same range is used for all dimensions that share the same
            unit. If set to `None`, ranges are determined automatically.
        :param smoothing: Standard deviation of the Gaussian kernel used to smooth the
            histograms.
        :param plot_1d_kws: Additional keyword arguments to be passed to
            `ParticleBeam.plot_1d_distribution` for plotting 1D histograms.
        :param plot_2d_kws: Additional keyword arguments to be passed to
            `ParticleBeam.plot_2d_distribution` for plotting 2D histograms.
        :param axs: List of Matplotlib axes objects to use for plotting. If set to
            `None`, a new figure is created. Must have the shape `(len(dimensions),
            len(dimensions))`.
        :return: Matplotlib figure and axes objects with the plot.
        """
        if axs is None:
            fig, axs = plt.subplots(
                len(dimensions),
                len(dimensions),
                figsize=(2 * len(dimensions), 2 * len(dimensions)),
            )
        else:
            fig = axs[0, 0].figure
            assert axs.shape == (len(dimensions), len(dimensions)), (
                "If `axs` is provided, it must have the shape "
                f"`({len(dimensions)}, {len(dimensions)})`."
            )

        # Determine bin ranges for all plots in the grid at once
        full_tensor = (
            torch.stack([getattr(self, dimension) for dimension in dimensions], dim=-2)
            .cpu()
            .detach()
            .numpy()
        )
        if bin_ranges is None:
            bin_ranges = [
                (
                    full_tensor[i, :].min()
                    - (full_tensor[i, :].max() - full_tensor[i, :].min()) / 10,
                    full_tensor[i, :].max()
                    + (full_tensor[i, :].max() - full_tensor[i, :].min()) / 10,
                )
                for i in range(full_tensor.shape[-2])
            ]
        if bin_ranges == "unit_same":
            spacial_idxs = [
                i
                for i, dimension in enumerate(dimensions)
                if dimension in ["x", "y", "tau"]
            ]
            spacial_bin_range = (
                full_tensor[spacial_idxs, :].min()
                - (
                    full_tensor[spacial_idxs, :].max()
                    - full_tensor[spacial_idxs, :].min()
                )
                / 10,
                full_tensor[spacial_idxs, :].max()
                + (
                    full_tensor[spacial_idxs, :].max()
                    - full_tensor[spacial_idxs, :].min()
                )
                / 10,
            )
            unitless_idxs = [
                i
                for i, dimension in enumerate(dimensions)
                if dimension in ["px", "py", "p"]
            ]
            unitless_bin_range = (
                full_tensor[unitless_idxs, :].min()
                - (
                    full_tensor[unitless_idxs, :].max()
                    - full_tensor[unitless_idxs, :].min()
                )
                / 10,
                full_tensor[unitless_idxs, :].max()
                + (
                    full_tensor[unitless_idxs, :].max()
                    - full_tensor[unitless_idxs, :].min()
                )
                / 10,
            )
            bin_range_dict = {
                "x": spacial_bin_range,
                "px": unitless_bin_range,
                "y": spacial_bin_range,
                "py": unitless_bin_range,
                "tau": spacial_bin_range,
                "p": unitless_bin_range,
            }
            bin_ranges = [bin_range_dict[dimension] for dimension in dimensions]
        if np.asarray(bin_ranges).shape == (2,):
            bin_ranges = [bin_ranges] * len(dimensions)
        assert len(bin_ranges) == len(dimensions) and all(
            len(e) == 2 for e in bin_ranges
        )

        # Plot diagonal 1D histograms on the diagonal
        diagonal_axs = [axs[i, i] for i, _ in enumerate(dimensions)]
        for dimension, bin_range, ax in zip(dimensions, bin_ranges, diagonal_axs):
            self.plot_1d_distribution(
                dimension=dimension,
                bins=bins,
                bin_range=bin_range,
                ax=ax,
                **(plot_1d_kws or {}),
            )

        # Plot 2D histograms on the off-diagonal
        for i, j in itertools.combinations(range(len(dimensions)), 2):
            self.plot_2d_distribution(
                x_dimension=dimensions[i],
                y_dimension=dimensions[j],
                bins=bins,
                bin_ranges=(bin_ranges[i], bin_ranges[j]),
                ax=axs[j, i],
                **(plot_2d_kws or {}),
            )

        # Hide unused axes
        for i, j in itertools.combinations(range(len(dimensions)), 2):
            axs[i, j].set_visible(False)

        # Clean up labels
        for ax_column in axs.T:
            for ax in ax_column[0:-1]:
                ax.sharex(ax_column[0])
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.set_xlabel(None)
        for i, ax_row in enumerate(axs):
            for ax in ax_row[1:i]:
                ax.sharey(ax_row[0])
                ax.yaxis.set_tick_params(labelleft=False)
                ax.set_ylabel(None)
        for i, _ in enumerate(dimensions):
            axs[i, i].sharey(axs[0, 0])
            axs[i, i].set_yticks([])
            axs[i, i].set_ylabel(None)

        return fig, axs

    def plot_point_cloud(
        self, scatter_kws: dict | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        """
        Plot a 3D point cloud of the spatial coordinates of the particles.

        :param scatter_kws: Additional keyword arguments to be passed to the `scatter`
            plotting function of matplotlib.
        :param ax: Matplotlib axes object to use for plotting.
        :return: Matplotlib axes object with the plot.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

        x = self.x.cpu().detach().numpy()
        tau = self.tau.cpu().detach().numpy()
        y = self.y.cpu().detach().numpy()

        ax.scatter(x, tau, y, c=self.p.cpu().detach().numpy(), **(scatter_kws or {}))
        ax.set_xlabel(f"{self.PRETTY_DIMENSION_LABELS['x']}")
        ax.set_ylabel(f"{self.PRETTY_DIMENSION_LABELS['tau']}")
        ax.set_zlabel(f"{self.PRETTY_DIMENSION_LABELS['y']}")

        # Handle units
        format_axis_with_prefixed_unit(ax.xaxis, "m", x)
        format_axis_with_prefixed_unit(ax.yaxis, "m", tau)
        format_axis_with_prefixed_unit(ax.zaxis, "m", y)

        return ax

    def __len__(self) -> int:
        return int(self.num_particles)

    @property
    def total_charge(self) -> torch.Tensor:
        """Total charge of the beam in C, taking into account particle losses."""
        return torch.sum(self.particle_charges * self.survival_probabilities, dim=-1)

    @property
    def num_particles(self) -> int:
        """
        Length of the macroparticle array.

        NOTE: This does not account for lost particles.
        """
        return self.particles.shape[-2]

    @property
    def num_particles_survived(self) -> torch.Tensor:
        """Number of macroparticles that have survived."""
        return self.survival_probabilities.sum(dim=-1)

    @property
    def x(self) -> torch.Tensor | None:
        return self.particles[..., 0]

    @x.setter
    def x(self, value: torch.Tensor) -> None:
        self.particles[..., 0] = value

    @property
    def mu_x(self) -> torch.Tensor | None:
        """
        Mean of the :math:`x` coordinates of the particles, weighted by their
        survival probability.
        """
        return torch.sum(
            (self.x * self.survival_probabilities), dim=-1
        ) / self.survival_probabilities.sum(dim=-1)

    @property
    def sigma_x(self) -> torch.Tensor | None:
        """
        Standard deviation of the :math:`x` coordinates of the particles, weighted
        by their survival probability.
        """
        return unbiased_weighted_std(
            self.x, weights=self.survival_probabilities, dim=-1
        )

    @property
    def px(self) -> torch.Tensor | None:
        return self.particles[..., 1]

    @px.setter
    def px(self, value: torch.Tensor) -> None:
        self.particles[..., 1] = value

    @property
    def mu_px(self) -> torch.Tensor | None:
        """
        Mean of the :math:`px` coordinates of the particles, weighted by their
        survival probability.
        """
        return torch.sum(
            (self.px * self.survival_probabilities), dim=-1
        ) / self.survival_probabilities.sum(dim=-1)

    @property
    def sigma_px(self) -> torch.Tensor | None:
        """
        Standard deviation of the :math:`px` coordinates of the particles, weighted
        by their survival probability.
        """
        return unbiased_weighted_std(
            self.px, weights=self.survival_probabilities, dim=-1
        )

    @property
    def y(self) -> torch.Tensor | None:
        return self.particles[..., 2]

    @y.setter
    def y(self, value: torch.Tensor) -> None:
        self.particles[..., 2] = value

    @property
    def mu_y(self) -> float | None:
        return torch.sum(
            (self.y * self.survival_probabilities), dim=-1
        ) / self.survival_probabilities.sum(dim=-1)

    @property
    def sigma_y(self) -> torch.Tensor | None:
        return unbiased_weighted_std(
            self.y, weights=self.survival_probabilities, dim=-1
        )

    @property
    def py(self) -> torch.Tensor | None:
        return self.particles[..., 3]

    @py.setter
    def py(self, value: torch.Tensor) -> None:
        self.particles[..., 3] = value

    @property
    def mu_py(self) -> torch.Tensor | None:
        return torch.sum(
            (self.py * self.survival_probabilities), dim=-1
        ) / self.survival_probabilities.sum(dim=-1)

    @property
    def sigma_py(self) -> torch.Tensor | None:
        return unbiased_weighted_std(
            self.py, weights=self.survival_probabilities, dim=-1
        )

    @property
    def tau(self) -> torch.Tensor | None:
        return self.particles[..., 4]

    @tau.setter
    def tau(self, value: torch.Tensor) -> None:
        self.particles[..., 4] = value

    @property
    def mu_tau(self) -> torch.Tensor | None:
        return torch.sum(
            (self.tau * self.survival_probabilities), dim=-1
        ) / self.survival_probabilities.sum(dim=-1)

    @property
    def sigma_tau(self) -> torch.Tensor | None:
        return unbiased_weighted_std(
            self.tau, weights=self.survival_probabilities, dim=-1
        )

    @property
    def p(self) -> torch.Tensor | None:
        return self.particles[..., 5]

    @p.setter
    def p(self, value: torch.Tensor) -> None:
        self.particles[..., 5] = value

    @property
    def mu_p(self) -> torch.Tensor | None:
        return torch.sum(
            (self.p * self.survival_probabilities), dim=-1
        ) / self.survival_probabilities.sum(dim=-1)

    @property
    def sigma_p(self) -> torch.Tensor | None:
        return unbiased_weighted_std(
            self.p, weights=self.survival_probabilities, dim=-1
        )

    @property
    def cov_xpx(self) -> torch.Tensor:
        r"""
        Returns the covariance between x and px. :math:`\sigma_{x, px}^2`.
        It is weighted by the survival probability of the particles.
        """
        return unbiased_weighted_covariance(
            self.x, self.px, weights=self.survival_probabilities, dim=-1
        )

    @property
    def cov_ypy(self) -> torch.Tensor:
        r"""
        Returns the covariance between y and py. :math:`\sigma_{y, py}^2`.
        It is weighted by the survival probability of the particles.
        """
        return unbiased_weighted_covariance(
            self.y, self.py, weights=self.survival_probabilities, dim=-1
        )

    @property
    def cov_taup(self) -> torch.Tensor:
        r"""
        Returns the covariance between tau and p. :math:`\sigma_{\tau, p}^2`.
        It is weighted by the survival probability of the particles.
        """
        return unbiased_weighted_covariance(
            self.tau, self.p, weights=self.survival_probabilities, dim=-1
        )

    @property
    def energies(self) -> torch.Tensor:
        """Energies of the individual particles."""
        return self.p * self.p0c + self.energy

    @property
    def momenta(self) -> torch.Tensor:
        """Momenta of the individual particles."""
        return torch.sqrt(self.energies**2 - self.species.mass_eV**2)

    def clone(self) -> "ParticleBeam":
        return self.__class__(
            particles=self.particles.clone(),
            energy=self.energy.clone(),
            particle_charges=self.particle_charges.clone(),
            survival_probabilities=self.survival_probabilities.clone(),
            s=self.s.clone(),
            species=self.species.clone(),
        )

    def __getitem__(self, item: int | slice | torch.Tensor) -> "ParticleBeam":
        vector_shape = torch.broadcast_shapes(
            self.particles.shape[:-2],
            self.energy.shape,
            self.particle_charges.shape[:-1],
            self.survival_probabilities.shape[:-1],
        )
        broadcasted_particles = torch.broadcast_to(
            self.particles, (*vector_shape, self.num_particles, 7)
        )
        broadcasted_energy = torch.broadcast_to(self.energy, vector_shape)
        broadcasted_particle_charges = torch.broadcast_to(
            self.particle_charges, (*vector_shape, self.num_particles)
        )
        broadcasted_survival_probabilities = torch.broadcast_to(
            self.survival_probabilities, (*vector_shape, self.num_particles)
        )

        return self.__class__(
            particles=broadcasted_particles[item],
            energy=broadcasted_energy[item],
            particle_charges=broadcasted_particle_charges[item],
            survival_probabilities=broadcasted_survival_probabilities[item],
            device=self.particles.device,
            dtype=self.particles.dtype,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(particles={self.particles}, "
            + f"energy={self.energy}, "
            + f"particle_charges={self.particle_charges}, "
            + f"survival_probabilities={self.survival_probabilities}, "
            + f"s={self.s}, "
            + f"species={repr(self.species)})"
        )
