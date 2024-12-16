from typing import Optional, Union

import numpy as np
import torch
from pmd_beamphysics import ParticleGroup
from scipy import constants
from scipy.constants import physical_constants
from torch.distributions import MultivariateNormal

from cheetah.particles.beam import Beam
from cheetah.utils import (
    elementwise_linspace,
    unbiased_weighted_covariance,
    unbiased_weighted_std,
    verify_device_and_dtype,
)

speed_of_light = torch.tensor(constants.speed_of_light)  # In m/s
electron_mass = torch.tensor(constants.electron_mass)  # In kg
electron_mass_eV = (
    physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
)  # In eV


class ParticleBeam(Beam):
    """
    Beam of charged particles, where each particle is simulated.

    :param particles: List of 7-dimensional particle vectors.
    :param energy: Reference energy of the beam in eV.
    :param particle_charges: Charges of the macroparticles in the beam in C.
    :param survival_probabilities: Vector of probabilities that each particle has
        survived (i.e. not been lost), where 1.0 means the particle has survived and
        0.0 means the particle has been lost. Defaults to ones.
    :param device: Device to move the beam's particle array to. If set to `"auto"` a
        CUDA GPU is selected if available. The CPU is used otherwise.
    :param dtype: Data type of the generated particles.
    """

    def __init__(
        self,
        particles: torch.Tensor,
        energy: torch.Tensor,
        particle_charges: Optional[torch.Tensor] = None,
        survival_probabilities: Optional[torch.Tensor] = None,
        device=None,
        dtype=None,
    ) -> None:
        device, dtype = verify_device_and_dtype(
            [particles, energy, particle_charges], device, dtype
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        assert (
            particles.shape[-2] > 0 and particles.shape[-1] == 7
        ), "Particle vectors must be 7-dimensional."

        self.register_buffer("particles", None)
        self.register_buffer("energy", None)
        self.register_buffer(
            "particle_charges", torch.zeros(particles.shape[-2], **factory_kwargs)
        )
        self.register_buffer(
            "survival_probabilities", torch.ones(particles.shape[-2], **factory_kwargs)
        )

        self.particles = particles.to(**factory_kwargs)
        self.energy = energy.to(**factory_kwargs)
        if particle_charges is not None:
            self.particle_charges = particle_charges.to(**factory_kwargs)
        if survival_probabilities is not None:
            self.survival_probabilities = survival_probabilities.to(**factory_kwargs)

    @classmethod
    def from_parameters(
        cls,
        num_particles: int = 100_000,
        mu_x: Optional[torch.Tensor] = None,
        mu_y: Optional[torch.Tensor] = None,
        mu_px: Optional[torch.Tensor] = None,
        mu_py: Optional[torch.Tensor] = None,
        sigma_x: Optional[torch.Tensor] = None,
        sigma_y: Optional[torch.Tensor] = None,
        sigma_px: Optional[torch.Tensor] = None,
        sigma_py: Optional[torch.Tensor] = None,
        sigma_tau: Optional[torch.Tensor] = None,
        sigma_p: Optional[torch.Tensor] = None,
        cor_x: Optional[torch.Tensor] = None,
        cor_y: Optional[torch.Tensor] = None,
        cor_tau: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        total_charge: Optional[torch.Tensor] = None,
        device=None,
        dtype=None,
    ) -> "ParticleBeam":
        """
        Generate Cheetah Beam of random particles.

        :param num_particles: Number of particles to generate.
        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_px: Center of the particle distribution on px, dimensionless.
        :param mu_py: Center of the particle distribution on py , dimensionless.
        :param sigma_x: Sigma of the particle distribution in x direction in meters.
        :param sigma_y: Sigma of the particle distribution in y direction in meters.
        :param sigma_px: Sigma of the particle distribution in px direction,
            dimensionless.
        :param sigma_py: Sigma of the particle distribution in py direction,
            dimensionless.
        :param sigma_tau: Sigma of the particle distribution in longitudinal direction,
            in meters.
        :param sigma_p: Sigma of the particle distribution in longitudinal momentum,
            dimensionless.
        :param cor_x: Correlation between x and px.
        :param cor_y: Correlation between y and py.
        :param cor_tau: Correlation between s and p.
        :param energy: Energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        :param device: Device to move the beam's particle array to. If set to `"auto"` a
            CUDA GPU is selected if available. The CPU is used otherwise.
        :param dtype: Data type of the generated particles.
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
                cor_x,
                cor_y,
                cor_tau,
                energy,
                total_charge,
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
        cor_x = cor_x if cor_x is not None else torch.tensor(0.0, **factory_kwargs)
        cor_y = cor_y if cor_y is not None else torch.tensor(0.0, **factory_kwargs)
        cor_tau = (
            cor_tau if cor_tau is not None else torch.tensor(0.0, **factory_kwargs)
        )
        energy = energy if energy is not None else torch.tensor(1e8, **factory_kwargs)
        total_charge = (
            total_charge
            if total_charge is not None
            else torch.tensor(0.0, **factory_kwargs)
        )
        particle_charges = (
            torch.ones((*total_charge.shape, num_particles), **factory_kwargs)
            * total_charge.unsqueeze(-1)
            / num_particles
        )

        mu_x, mu_px, mu_y, mu_py = torch.broadcast_tensors(mu_x, mu_px, mu_y, mu_py)
        mean = torch.stack(
            [mu_x, mu_px, mu_y, mu_py, torch.zeros_like(mu_x), torch.zeros_like(mu_x)],
            dim=-1,
        )

        (
            sigma_x,
            cor_x,
            sigma_px,
            sigma_y,
            cor_y,
            sigma_py,
            sigma_tau,
            cor_tau,
            sigma_p,
        ) = torch.broadcast_tensors(
            sigma_x,
            cor_x,
            sigma_px,
            sigma_y,
            cor_y,
            sigma_py,
            sigma_tau,
            cor_tau,
            sigma_p,
        )
        cov = torch.zeros(*sigma_x.shape, 6, 6, **factory_kwargs)
        cov[..., 0, 0] = sigma_x**2
        cov[..., 0, 1] = cor_x
        cov[..., 1, 0] = cor_x
        cov[..., 1, 1] = sigma_px**2
        cov[..., 2, 2] = sigma_y**2
        cov[..., 2, 3] = cor_y
        cov[..., 3, 2] = cor_y
        cov[..., 3, 3] = sigma_py**2
        cov[..., 4, 4] = sigma_tau**2
        cov[..., 4, 5] = cor_tau
        cov[..., 5, 4] = cor_tau
        cov[..., 5, 5] = sigma_p**2

        vector_shape = torch.broadcast_shapes(mean.shape[:-1], cov.shape[:-2])
        mean = mean.expand(*vector_shape, 6)
        cov = cov.expand(*vector_shape, 6, 6)
        particles = torch.ones((*vector_shape, num_particles, 7), **factory_kwargs)
        distributions = [
            MultivariateNormal(sample_mean, covariance_matrix=sample_cov)
            for sample_mean, sample_cov in zip(mean.view(-1, 6), cov.view(-1, 6, 6))
        ]
        particles[..., :6] = torch.stack(
            [distribution.sample((num_particles,)) for distribution in distributions],
            dim=0,
        ).view(*vector_shape, num_particles, 6)

        return cls(
            particles,
            energy,
            particle_charges=particle_charges,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_twiss(
        cls,
        num_particles: int = 100_000,
        beta_x: Optional[torch.Tensor] = None,
        alpha_x: Optional[torch.Tensor] = None,
        emittance_x: Optional[torch.Tensor] = None,
        beta_y: Optional[torch.Tensor] = None,
        alpha_y: Optional[torch.Tensor] = None,
        emittance_y: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        sigma_tau: Optional[torch.Tensor] = None,
        sigma_p: Optional[torch.Tensor] = None,
        cor_tau: Optional[torch.Tensor] = None,
        total_charge: Optional[torch.Tensor] = None,
        device=None,
        dtype=None,
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
                cor_tau,
                total_charge,
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
        cor_tau = (
            cor_tau if cor_tau is not None else torch.tensor(0.0, **factory_kwargs)
        )
        total_charge = (
            total_charge
            if total_charge is not None
            else torch.tensor(0.0, **factory_kwargs)
        )

        sigma_x = torch.sqrt(beta_x * emittance_x)
        sigma_px = torch.sqrt(emittance_x * (1 + alpha_x**2) / beta_x)
        sigma_y = torch.sqrt(beta_y * emittance_y)
        sigma_py = torch.sqrt(emittance_y * (1 + alpha_y**2) / beta_y)
        cor_x = -emittance_x * alpha_x
        cor_y = -emittance_y * alpha_y

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
            cor_tau=cor_tau,
            cor_x=cor_x,
            cor_y=cor_y,
            total_charge=total_charge,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def uniform_3d_ellipsoid(
        cls,
        num_particles: int = 100_000,
        radius_x: Optional[torch.Tensor] = None,
        radius_y: Optional[torch.Tensor] = None,
        radius_tau: Optional[torch.Tensor] = None,
        sigma_px: Optional[torch.Tensor] = None,
        sigma_py: Optional[torch.Tensor] = None,
        sigma_p: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        total_charge: Optional[torch.Tensor] = None,
        device=None,
        dtype=None,
    ):
        """
        Generate a particle beam with spatially uniformly distributed particles inside
        an ellipsoid, i.e. a waterbag distribution.

        Note that:
         - The generated particles do not have correlation in the momentum directions,
           and by default a cold beam with no divergence is generated.
         - For vectorised generation, parameters that are not `None` must have the same
           shape.

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

        # Generate x, y and tau within the ellipsoid
        # Broadcasting with (1,) is a hack to make the loop work. Interestingly it
        # this does not break the assigments into the non-vectorised particle tensor of
        # the beam object.
        vector_shape = torch.broadcast_shapes(
            radius_x.shape, radius_y.shape, radius_tau.shape, (1,)
        )
        flattened_x = torch.empty(
            *vector_shape, num_particles, **factory_kwargs
        ).flatten(end_dim=-2)
        flattened_y = torch.empty(
            *vector_shape, num_particles, **factory_kwargs
        ).flatten(end_dim=-2)
        flattened_tau = torch.empty(
            *vector_shape, num_particles, **factory_kwargs
        ).flatten(end_dim=-2)
        for i, (r_x, r_y, r_tau) in enumerate(
            zip(radius_x.flatten(), radius_y.flatten(), radius_tau.flatten())
        ):
            num_successful = 0
            while num_successful < num_particles:
                x = (torch.rand(num_particles, **factory_kwargs) - 0.5) * 2 * r_x
                y = (torch.rand(num_particles, **factory_kwargs) - 0.5) * 2 * r_y
                tau = (torch.rand(num_particles, **factory_kwargs) - 0.5) * 2 * r_tau

                is_in_ellipsoid = x**2 / r_x**2 + y**2 / r_y**2 + tau**2 / r_tau**2 < 1
                num_to_add = min(num_particles - num_successful, is_in_ellipsoid.sum())

                flattened_x[i, num_successful : num_successful + num_to_add] = x[
                    is_in_ellipsoid
                ][:num_to_add]
                flattened_y[i, num_successful : num_successful + num_to_add] = y[
                    is_in_ellipsoid
                ][:num_to_add]
                flattened_tau[i, num_successful : num_successful + num_to_add] = tau[
                    is_in_ellipsoid
                ][:num_to_add]

                num_successful += num_to_add

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
            device=device,
            dtype=dtype,
        )

        # Replace the spatial coordinates with the generated ones
        beam.x = flattened_x.view(*vector_shape, num_particles)
        beam.y = flattened_y.view(*vector_shape, num_particles)
        beam.tau = flattened_tau.view(*vector_shape, num_particles)

        return beam

    @classmethod
    def make_linspaced(
        cls,
        num_particles: int = 10,
        mu_x: Optional[torch.Tensor] = None,
        mu_y: Optional[torch.Tensor] = None,
        mu_px: Optional[torch.Tensor] = None,
        mu_py: Optional[torch.Tensor] = None,
        sigma_x: Optional[torch.Tensor] = None,
        sigma_y: Optional[torch.Tensor] = None,
        sigma_px: Optional[torch.Tensor] = None,
        sigma_py: Optional[torch.Tensor] = None,
        sigma_tau: Optional[torch.Tensor] = None,
        sigma_p: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        total_charge: Optional[torch.Tensor] = None,
        device=None,
        dtype=None,
    ) -> "ParticleBeam":
        """
        Generate Cheetah Beam of *n* linspaced particles.

        :param n: Number of particles to generate.
        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_px: Center of the particle distribution on px, dimensionless.
        :param mu_py: Center of the particle distribution on py , dimensionless.
        :param sigma_x: Sigma of the particle distribution in x direction in meters.
        :param sigma_y: Sigma of the particle distribution in y direction in meters.
        :param sigma_px: Sigma of the particle distribution in px direction,
            dimensionless.
        :param sigma_py: Sigma of the particle distribution in py direction,
            dimensionless.
        :param sigma_tau: Sigma of the particle distribution in longitudinal direction,
            in meters.
        :param sigma_p: Sigma of the particle distribution in p, dimensionless.
        :param energy: Energy of the beam in eV.
        :param device: Device to move the beam's particle array to. If set to `"auto"` a
            CUDA GPU is selected if available. The CPU is used otherwise.
        :param dtype: Data type of the generated particles.
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
            else torch.tensor(0.0, **factory_kwargs)
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
        particles[..., 4] = elementwise_linspace(-sigma_tau, sigma_tau, num_particles)
        particles[..., 5] = elementwise_linspace(-sigma_p, sigma_p, num_particles)

        return cls(
            particles=particles,
            energy=energy,
            particle_charges=particle_charges,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_ocelot(cls, parray, device=None, dtype=torch.float32) -> "ParticleBeam":
        """
        Convert an Ocelot ParticleArray `parray` to a Cheetah Beam.
        """
        num_particles = parray.rparticles.shape[1]
        particles = torch.ones((num_particles, 7))
        particles[:, :6] = torch.tensor(parray.rparticles.transpose())
        particle_charges = torch.tensor(parray.q_array)

        return cls(
            particles=particles.unsqueeze(0),
            energy=torch.tensor(1e9 * parray.E).unsqueeze(0),
            particle_charges=particle_charges.unsqueeze(0),
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_astra(cls, path: str, device=None, dtype=torch.float32) -> "ParticleBeam":
        """Load an Astra particle distribution as a Cheetah Beam."""
        from cheetah.converters.astra import from_astrabeam

        particles, energy, particle_charges = from_astrabeam(path)
        particles_7d = torch.ones((particles.shape[0], 7))
        particles_7d[:, :6] = torch.from_numpy(particles)
        particle_charges = torch.from_numpy(particle_charges)
        return cls(
            particles=particles_7d,
            energy=torch.tensor(energy),
            particle_charges=particle_charges,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_openpmd_file(
        cls, path: str, energy: torch.Tensor, device=None, dtype=torch.float32
    ) -> "ParticleBeam":
        """Load an OpenPMD particle distribution file as a Cheetah Beam."""

        particle_group = ParticleGroup(path)
        return cls.from_openpmd_particlegroup(
            particle_group, energy, device=device, dtype=dtype
        )

    @classmethod
    def from_openpmd_particlegroup(
        cls,
        particle_group: ParticleGroup,
        energy: torch.Tensor,
        device=None,
        dtype=torch.float32,
    ) -> "ParticleBeam":
        """Convert an OpenPMD ParticleGroup to a Cheetah Beam.

        :param particle_group: OpenPMD ParticleGroup object.
        :param energy: Reference energy of the beam in eV.
        :param device: Device to move the beam's particle array to. If set to `"auto"` a
            CUDA GPU is selected if available. The CPU is used otherwise.
        :param dtype: Data type of the generated particles.
        """

        # Assume only electron now
        p0c = torch.sqrt(energy**2 - electron_mass_eV**2)

        x = torch.from_numpy(particle_group.x)
        y = torch.from_numpy(particle_group.y)
        px = torch.from_numpy(particle_group.px) / p0c
        py = torch.from_numpy(particle_group.py) / p0c
        tau = torch.from_numpy(particle_group.t) * speed_of_light
        delta = (torch.from_numpy(particle_group.energy) - energy) / p0c

        particles = torch.stack([x, px, y, py, tau, delta, torch.ones_like(x)], dim=-1)
        particle_charges = torch.from_numpy(particle_group.weight)
        survival_probabilities = torch.from_numpy(particle_group.status)

        return cls(
            particles=particles,
            energy=energy,
            particle_charges=particle_charges,
            survival_probabilities=survival_probabilities,
            device=device,
            dtype=dtype,
        )

    def transformed_to(
        self,
        mu_x: Optional[torch.Tensor] = None,
        mu_y: Optional[torch.Tensor] = None,
        mu_px: Optional[torch.Tensor] = None,
        mu_py: Optional[torch.Tensor] = None,
        sigma_x: Optional[torch.Tensor] = None,
        sigma_y: Optional[torch.Tensor] = None,
        sigma_px: Optional[torch.Tensor] = None,
        sigma_py: Optional[torch.Tensor] = None,
        sigma_tau: Optional[torch.Tensor] = None,
        sigma_p: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        total_charge: Optional[torch.Tensor] = None,
        device=None,
        dtype=None,
    ) -> "ParticleBeam":
        """
        Create version of this beam that is transformed to new beam parameters.

        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_px: Center of the particle distribution on px, dimensionless.
        :param mu_py: Center of the particle distribution on py , dimensionless.
        :param sigma_x: Sigma of the particle distribution in x direction in meters.
        :param sigma_y: Sigma of the particle distribution in y direction in meters.
        :param sigma_px: Sigma of the particle distribution in px direction,
            dimensionless.
        :param sigma_py: Sigma of the particle distribution in py direction,
            dimensionless.
        :param sigma_tau: Sigma of the particle distribution in longitudinal direction,
            in meters.
        :param sigma_p: Sigma of the particle distribution in p, dimensionless.
        :param energy: Reference energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        :param device: Device to move the beam's particle array to. If set to `"auto"` a
            CUDA GPU is selected if available. The CPU is used otherwise.
        :param dtype: Data type of the transformed particles.
        """
        device = device if device is not None else self.mu_x.device
        dtype = dtype if dtype is not None else self.mu_x.dtype

        mu_x = mu_x if mu_x is not None else self.mu_x
        mu_y = mu_y if mu_y is not None else self.mu_y
        mu_px = mu_px if mu_px is not None else self.mu_px
        mu_py = mu_py if mu_py is not None else self.mu_py
        sigma_x = sigma_x if sigma_x is not None else self.sigma_x
        sigma_y = sigma_y if sigma_y is not None else self.sigma_y
        sigma_px = sigma_px if sigma_px is not None else self.sigma_px
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

        mu_x, mu_px, mu_y, mu_py = torch.broadcast_tensors(mu_x, mu_px, mu_y, mu_py)
        new_mu = torch.stack(
            [mu_x, mu_px, mu_y, mu_py, torch.zeros_like(mu_x), torch.zeros_like(mu_x)],
            dim=-1,
        )
        sigma_x, sigma_px, sigma_y, sigma_py, sigma_tau, sigma_p = (
            torch.broadcast_tensors(
                sigma_x, sigma_px, sigma_y, sigma_py, sigma_tau, sigma_p
            )
        )
        new_sigma = torch.stack(
            [sigma_x, sigma_px, sigma_y, sigma_py, sigma_tau, sigma_p], dim=-1
        )

        old_mu = torch.stack(
            [
                self.mu_x,
                self.mu_px,
                self.mu_y,
                self.mu_py,
                torch.zeros_like(self.mu_x),
                torch.zeros_like(self.mu_x),
            ],
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
            device=device,
            dtype=dtype,
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
            mu_y=self.mu_y,
            mu_px=self.mu_px,
            mu_py=self.mu_py,
            sigma_x=self.sigma_x,
            sigma_y=self.sigma_y,
            sigma_px=self.sigma_px,
            sigma_py=self.sigma_py,
            sigma_tau=self.sigma_tau,
            sigma_p=self.sigma_p,
            energy=self.energy,
            total_charge=self.total_charge,
            device=self.particles.device,
            dtype=self.particles.dtype,
        )

    @classmethod
    def from_xyz_pxpypz(
        cls,
        xp_coordinates: torch.Tensor,
        energy: torch.Tensor,
        particle_charges: Optional[torch.Tensor] = None,
        survival_probabilities: Optional[torch.Tensor] = None,
        device=None,
        dtype=torch.float32,
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
            device=device,
            dtype=dtype,
        )

        p0 = (
            beam.relativistic_gamma
            * beam.relativistic_beta
            * electron_mass
            * speed_of_light
        )
        p = torch.sqrt(
            xp_coordinates[..., 1] ** 2
            + xp_coordinates[..., 3] ** 2
            + xp_coordinates[..., 5] ** 2
        )
        gamma = torch.sqrt(1 + (p / (electron_mass * speed_of_light)) ** 2)

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
            * electron_mass
            * speed_of_light
        )  # Reference momentum in (kg m/s)
        gamma = self.relativistic_gamma.unsqueeze(-1) * (
            torch.ones(self.particles.shape[:-1])
            + self.particles[..., 5] * self.relativistic_beta.unsqueeze(-1)
        )
        beta = torch.sqrt(1 - 1 / gamma**2)
        momentum = gamma * electron_mass * beta * speed_of_light

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
    def x(self) -> Optional[torch.Tensor]:
        return self.particles[..., 0]

    @x.setter
    def x(self, value: torch.Tensor) -> None:
        self.particles[..., 0] = value

    @property
    def mu_x(self) -> Optional[torch.Tensor]:
        """
        Mean of the :math:`x` coordinates of the particles, weighted by their
        survival probability.
        """
        return torch.sum(
            (self.x * self.survival_probabilities), dim=-1
        ) / self.survival_probabilities.sum(dim=-1)

    @property
    def sigma_x(self) -> Optional[torch.Tensor]:
        """
        Standard deviation of the :math:`x` coordinates of the particles, weighted
        by their survival probability.
        """
        return unbiased_weighted_std(
            self.x, weights=self.survival_probabilities, dim=-1
        )

    @property
    def px(self) -> Optional[torch.Tensor]:
        return self.particles[..., 1]

    @px.setter
    def px(self, value: torch.Tensor) -> None:
        self.particles[..., 1] = value

    @property
    def mu_px(self) -> Optional[torch.Tensor]:
        """
        Mean of the :math:`px` coordinates of the particles, weighted by their
        survival probability.
        """
        return torch.sum(
            (self.px * self.survival_probabilities), dim=-1
        ) / self.survival_probabilities.sum(dim=-1)

    @property
    def sigma_px(self) -> Optional[torch.Tensor]:
        """
        Standard deviation of the :math:`px` coordinates of the particles, weighted
        by their survival probability.
        """
        return unbiased_weighted_std(
            self.px, weights=self.survival_probabilities, dim=-1
        )

    @property
    def y(self) -> Optional[torch.Tensor]:
        return self.particles[..., 2]

    @y.setter
    def y(self, value: torch.Tensor) -> None:
        self.particles[..., 2] = value

    @property
    def mu_y(self) -> Optional[float]:
        return torch.sum(
            (self.y * self.survival_probabilities), dim=-1
        ) / self.survival_probabilities.sum(dim=-1)

    @property
    def sigma_y(self) -> Optional[torch.Tensor]:
        return unbiased_weighted_std(
            self.y, weights=self.survival_probabilities, dim=-1
        )

    @property
    def py(self) -> Optional[torch.Tensor]:
        return self.particles[..., 3]

    @py.setter
    def py(self, value: torch.Tensor) -> None:
        self.particles[..., 3] = value

    @property
    def mu_py(self) -> Optional[torch.Tensor]:
        return torch.sum(
            (self.py * self.survival_probabilities), dim=-1
        ) / self.survival_probabilities.sum(dim=-1)

    @property
    def sigma_py(self) -> Optional[torch.Tensor]:
        return unbiased_weighted_std(
            self.py, weights=self.survival_probabilities, dim=-1
        )

    @property
    def tau(self) -> Optional[torch.Tensor]:
        return self.particles[..., 4]

    @tau.setter
    def tau(self, value: torch.Tensor) -> None:
        self.particles[..., 4] = value

    @property
    def mu_tau(self) -> Optional[torch.Tensor]:
        return torch.sum(
            (self.tau * self.survival_probabilities), dim=-1
        ) / self.survival_probabilities.sum(dim=-1)

    @property
    def sigma_tau(self) -> Optional[torch.Tensor]:
        return unbiased_weighted_std(
            self.tau, weights=self.survival_probabilities, dim=-1
        )

    @property
    def p(self) -> Optional[torch.Tensor]:
        return self.particles[..., 5]

    @p.setter
    def p(self, value: torch.Tensor) -> None:
        self.particles[..., 5] = value

    @property
    def mu_p(self) -> Optional[torch.Tensor]:
        return torch.sum(
            (self.p * self.survival_probabilities), dim=-1
        ) / self.survival_probabilities.sum(dim=-1)

    @property
    def sigma_p(self) -> Optional[torch.Tensor]:
        return unbiased_weighted_std(
            self.p, weights=self.survival_probabilities, dim=-1
        )

    @property
    def sigma_xpx(self) -> torch.Tensor:
        r"""
        Returns the covariance between x and px. :math:`\sigma_{x, px}^2`.
        It is weighted by the survival probability of the particles.
        """
        return unbiased_weighted_covariance(
            self.x, self.px, weights=self.survival_probabilities, dim=-1
        )

    @property
    def sigma_ypy(self) -> torch.Tensor:
        r"""
        Returns the covariance between y and py. :math:`\sigma_{y, py}^2`.
        It is weighted by the survival probability of the particles.
        """
        return unbiased_weighted_covariance(
            self.y, self.py, weights=self.survival_probabilities, dim=-1
        )

    @property
    def energies(self) -> torch.Tensor:
        """Energies of the individual particles."""
        return self.p * self.p0c + self.energy

    @property
    def momenta(self) -> torch.Tensor:
        """Momenta of the individual particles."""
        return torch.sqrt(self.energies**2 - electron_mass_eV**2)

    def save_as_openpmd_h5(self, filename: str) -> None:
        """Save the ParticleBeam as an OpenPMD beamphysics HDF5 file.

        :param filename: Path to the file where the beam should be saved.
        """
        particle_group = self.to_openpmd_particlegroup()
        particle_group.write(filename)

    def to_openpmd_particlegroup(self) -> None:
        """
        Convert the beam to an OpenPMD-beamphysics ParticleGroup object.

        Note: Currently OpenPMD-beamphysics only supports boolean particle status
        flags, i.e. alive or dead. The survival_probabilities will be converted to
        status flags by thresholding at 0.5.

        :return: OpenPMD-beamphysics ParticleGroup object with the beam's particles.
        """
        # For now only support none-batched particles
        if len(self.particles.shape) != 2:
            raise ValueError("Only non-batched particles are supported.")

        n_particles = self.num_particles
        weights = np.ones(n_particles)
        px = self.px * self.p0c
        py = self.py * self.p0c
        p_total = torch.sqrt(self.energies**2 - electron_mass_eV**2)
        pz = torch.sqrt(p_total**2 - px**2 - py**2)
        t = self.tau / speed_of_light
        weights = self.particle_charges
        # To be discussed
        status = self.survival_probabilities > 0.5

        data = {
            "x": self.x.numpy(),
            "y": self.y.numpy(),
            "z": self.tau.numpy(),
            "px": px.numpy(),
            "py": py.numpy(),
            "pz": pz.numpy(),
            "t": t.numpy(),
            "weight": weights,
            "status": status,
            # To be modified after adding support for other species
            "species": "electron",
        }
        particle_group = ParticleGroup(data=data)

        return particle_group

    def clone(self) -> "ParticleBeam":
        return ParticleBeam(
            particles=self.particles.clone(),
            energy=self.energy.clone(),
            particle_charges=self.particle_charges.clone(),
            survival_probabilities=self.survival_probabilities.clone(),
        )

    def __getitem__(self, item: Union[int, slice, torch.Tensor]) -> "ParticleBeam":
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
            f"{self.__class__.__name__}(n={repr(self.num_particles)},"
            f" mu_x={repr(self.mu_x)}, mu_px={repr(self.mu_px)},"
            f" mu_y={repr(self.mu_y)}, mu_py={repr(self.mu_py)},"
            f" sigma_x={repr(self.sigma_x)}, sigma_px={repr(self.sigma_px)},"
            f" sigma_y={repr(self.sigma_y)}, sigma_py={repr(self.sigma_py)},"
            f" sigma_tau={repr(self.sigma_tau)}, sigma_p={repr(self.sigma_p)},"
            f" energy={repr(self.energy)})"
            f" total_charge={repr(self.total_charge)})"
        )
