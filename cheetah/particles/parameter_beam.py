from typing import Optional

import numpy as np
import torch

from cheetah.particles.beam import Beam
from cheetah.particles.particle_beam import ParticleBeam
from cheetah.utils import verify_device_and_dtype


class ParameterBeam(Beam):
    """
    Beam of charged particles, where each particle is simulated.

    :param mu: Mu vector of the beam with shape `(..., 7)`.
    :param cov: Covariance matrix of the beam with shape `(..., 7, 7)`.
    :param energy: Reference energy of the beam in eV.
    :param total_charge: Total charge of the beam in C.
    :param device: Device to use for the beam. If "auto", use CUDA if available.
        Note: Compuationally it would be faster to use CPU for ParameterBeam.
    """

    def __init__(
        self,
        mu: torch.Tensor,
        cov: torch.Tensor,
        energy: torch.Tensor,
        total_charge: Optional[torch.Tensor] = None,
        device=None,
        dtype=None,
    ) -> None:
        device, dtype = verify_device_and_dtype(
            [mu, cov, energy, total_charge], device, dtype
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.register_buffer("_mu", None)
        self.register_buffer("_cov", None)
        self.register_buffer("energy", None)
        self.register_buffer("total_charge", torch.tensor(0.0, **factory_kwargs))

        self._mu = torch.as_tensor(mu, **factory_kwargs)
        self._cov = torch.as_tensor(cov, **factory_kwargs)
        self.energy = torch.as_tensor(energy, **factory_kwargs)
        if total_charge is not None:
            self.total_charge = torch.as_tensor(total_charge, **factory_kwargs)

    @classmethod
    def from_parameters(
        cls,
        mu_x: Optional[torch.Tensor] = None,
        mu_px: Optional[torch.Tensor] = None,
        mu_y: Optional[torch.Tensor] = None,
        mu_py: Optional[torch.Tensor] = None,
        sigma_x: Optional[torch.Tensor] = None,
        sigma_px: Optional[torch.Tensor] = None,
        sigma_y: Optional[torch.Tensor] = None,
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
    ) -> "ParameterBeam":
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
        mu_x = (
            torch.as_tensor(mu_x, **factory_kwargs)
            if mu_x is not None
            else torch.tensor(0.0, **factory_kwargs)
        )
        mu_px = (
            torch.as_tensor(mu_px, **factory_kwargs)
            if mu_px is not None
            else torch.tensor(0.0, **factory_kwargs)
        )
        mu_y = (
            torch.as_tensor(mu_y, **factory_kwargs)
            if mu_y is not None
            else torch.tensor(0.0, **factory_kwargs)
        )
        mu_py = (
            torch.as_tensor(mu_py, **factory_kwargs)
            if mu_py is not None
            else torch.tensor(0.0, **factory_kwargs)
        )
        sigma_x = (
            torch.as_tensor(sigma_x, **factory_kwargs)
            if sigma_x is not None
            else torch.tensor(175e-9, **factory_kwargs)
        )
        sigma_px = (
            torch.as_tensor(sigma_px, **factory_kwargs)
            if sigma_px is not None
            else torch.tensor(2e-7, **factory_kwargs)
        )
        sigma_y = (
            torch.as_tensor(sigma_y, **factory_kwargs)
            if sigma_y is not None
            else torch.tensor(175e-9, **factory_kwargs)
        )
        sigma_py = (
            torch.as_tensor(sigma_py, **factory_kwargs)
            if sigma_py is not None
            else torch.tensor(2e-7, **factory_kwargs)
        )
        sigma_tau = (
            torch.as_tensor(sigma_tau, **factory_kwargs)
            if sigma_tau is not None
            else torch.tensor(1e-6, **factory_kwargs)
        )
        sigma_p = (
            torch.as_tensor(sigma_p, **factory_kwargs)
            if sigma_p is not None
            else torch.tensor(1e-6, **factory_kwargs)
        )
        cor_x = (
            torch.as_tensor(cor_x, **factory_kwargs)
            if cor_x is not None
            else torch.tensor(0.0, **factory_kwargs)
        )
        cor_y = (
            torch.as_tensor(cor_y, **factory_kwargs)
            if cor_y is not None
            else torch.tensor(0.0, **factory_kwargs)
        )
        cor_tau = (
            torch.as_tensor(cor_tau, **factory_kwargs)
            if cor_tau is not None
            else torch.tensor(0.0, **factory_kwargs)
        )
        energy = (
            torch.as_tensor(energy, **factory_kwargs)
            if energy is not None
            else torch.tensor(1e8, **factory_kwargs)
        )
        total_charge = (
            torch.as_tensor(total_charge, **factory_kwargs)
            if total_charge is not None
            else torch.tensor(0.0, **factory_kwargs)
        )

        mu_x, mu_px, mu_y, mu_py = torch.broadcast_tensors(mu_x, mu_px, mu_y, mu_py)
        mu = torch.stack(
            [
                mu_x,
                mu_px,
                mu_y,
                mu_py,
                torch.zeros_like(mu_x),
                torch.zeros_like(mu_x),
                torch.ones_like(mu_x),
            ],
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
        cov = torch.zeros(*sigma_x.shape, 7, 7, **factory_kwargs)
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

        return cls(
            mu=mu,
            cov=cov,
            energy=energy,
            total_charge=total_charge,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_twiss(
        cls,
        beta_x: Optional[torch.Tensor] = None,
        alpha_x: Optional[torch.Tensor] = None,
        emittance_x: Optional[torch.Tensor] = None,
        beta_y: Optional[torch.Tensor] = None,
        alpha_y: Optional[torch.Tensor] = None,
        emittance_y: Optional[torch.Tensor] = None,
        sigma_tau: Optional[torch.Tensor] = None,
        sigma_p: Optional[torch.Tensor] = None,
        cor_tau: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        total_charge: Optional[torch.Tensor] = None,
        device=None,
        dtype=None,
    ) -> "ParameterBeam":
        # Extract device and dtype from given arguments
        device, dtype = verify_device_and_dtype(
            [
                beta_x,
                alpha_x,
                emittance_x,
                beta_y,
                alpha_y,
                emittance_y,
                sigma_tau,
                sigma_p,
                cor_tau,
                energy,
                total_charge,
            ],
            device,
            dtype,
        )
        factory_kwargs = {"device": device, "dtype": dtype}

        # Set default values without function call in function signature
        beta_x = (
            torch.as_tensor(beta_x, **factory_kwargs)
            if beta_x is not None
            else torch.tensor(1.0, **factory_kwargs)
        )
        alpha_x = (
            torch.as_tensor(alpha_x, **factory_kwargs)
            if alpha_x is not None
            else torch.tensor(0.0, **factory_kwargs)
        )
        emittance_x = (
            torch.as_tensor(emittance_x, **factory_kwargs)
            if emittance_x is not None
            else torch.tensor(7.1971891e-13, **factory_kwargs)
        )
        beta_y = (
            torch.as_tensor(beta_y, **factory_kwargs)
            if beta_y is not None
            else torch.tensor(1.0, **factory_kwargs)
        )
        alpha_y = (
            torch.as_tensor(alpha_y, **factory_kwargs)
            if alpha_y is not None
            else torch.tensor(0.0, **factory_kwargs)
        )
        emittance_y = (
            torch.as_tensor(emittance_y, **factory_kwargs)
            if emittance_y is not None
            else torch.tensor(7.1971891e-13, **factory_kwargs)
        )
        sigma_tau = (
            torch.as_tensor(sigma_tau, **factory_kwargs)
            if sigma_tau is not None
            else torch.tensor(1e-6, **factory_kwargs)
        )
        sigma_p = (
            torch.as_tensor(sigma_p, **factory_kwargs)
            if sigma_p is not None
            else torch.tensor(1e-6, **factory_kwargs)
        )
        cor_tau = (
            torch.as_tensor(cor_tau, **factory_kwargs)
            if cor_tau is not None
            else torch.tensor(0.0, **factory_kwargs)
        )
        energy = (
            torch.as_tensor(energy, **factory_kwargs)
            if energy is not None
            else torch.tensor(1e8, **factory_kwargs)
        )
        total_charge = (
            torch.as_tensor(total_charge, **factory_kwargs)
            if total_charge is not None
            else torch.tensor(0.0, **factory_kwargs)
        )

        assert torch.all(
            beta_x > 0
        ), "Beta function in x direction must be larger than 0 everywhere."
        assert torch.all(
            beta_y > 0
        ), "Beta function in y direction must be larger than 0 everywhere."

        sigma_x = torch.sqrt(emittance_x * beta_x)
        sigma_px = torch.sqrt(emittance_x * (1 + alpha_x**2) / beta_x)
        sigma_y = torch.sqrt(emittance_y * beta_y)
        sigma_py = torch.sqrt(emittance_y * (1 + alpha_y**2) / beta_y)
        cor_x = -emittance_x * alpha_x
        cor_y = -emittance_y * alpha_y
        return cls.from_parameters(
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
    def from_ocelot(cls, parray, device=None, dtype=torch.float32) -> "ParameterBeam":
        """Load an Ocelot ParticleArray `parray` as a Cheetah Beam."""
        mu = torch.ones(7)
        mu[:6] = torch.tensor(parray.rparticles.mean(axis=1), dtype=torch.float32)

        cov = torch.zeros(7, 7)
        cov[:6, :6] = torch.tensor(np.cov(parray.rparticles), dtype=torch.float32)

        energy = torch.tensor(1e9 * parray.E, dtype=torch.float32)
        total_charge = torch.tensor(np.sum(parray.q_array), dtype=torch.float32)

        return cls(
            mu=mu.unsqueeze(0),
            cov=cov.unsqueeze(0),
            energy=energy.unsqueeze(0),
            total_charge=total_charge.unsqueeze(0),
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_astra(cls, path: str, device=None, dtype=torch.float32) -> "ParameterBeam":
        """Load an Astra particle distribution as a Cheetah Beam."""
        from cheetah.converters.astra import from_astrabeam

        particles, energy, particle_charges = from_astrabeam(path)
        mu = torch.ones(7)
        mu[:6] = torch.tensor(particles.mean(axis=0))

        cov = torch.zeros(7, 7)
        cov[:6, :6] = torch.tensor(np.cov(particles.transpose()), dtype=torch.float32)

        total_charge = torch.tensor(np.sum(particle_charges), dtype=torch.float32)

        return cls(
            mu=mu,
            cov=cov,
            energy=torch.tensor(energy, dtype=torch.float32),
            total_charge=total_charge,
            device=device,
            dtype=dtype,
        )

    def transformed_to(
        self,
        mu_x: Optional[torch.Tensor] = None,
        mu_px: Optional[torch.Tensor] = None,
        mu_y: Optional[torch.Tensor] = None,
        mu_py: Optional[torch.Tensor] = None,
        sigma_x: Optional[torch.Tensor] = None,
        sigma_px: Optional[torch.Tensor] = None,
        sigma_y: Optional[torch.Tensor] = None,
        sigma_py: Optional[torch.Tensor] = None,
        sigma_tau: Optional[torch.Tensor] = None,
        sigma_p: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        total_charge: Optional[torch.Tensor] = None,
        device=None,
        dtype=None,
    ) -> "ParameterBeam":
        """
        Create version of this beam that is transformed to new beam parameters.

        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_px: Center of the particle distribution on px, dimensionless.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_py: Center of the particle distribution on yp, dimensionless.
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
        :param device: Device to create the transformed beam on. If set to `"auto"` a
            CUDA GPU is selected if available. The CPU is used otherwise.
        :param dtype: Data type of the transformed beam.
        """
        device = device if device is not None else self.mu_x.device
        dtype = dtype if dtype is not None else self.mu_x.dtype

        mu_x = mu_x if mu_x is not None else self.mu_x
        mu_px = mu_px if mu_px is not None else self.mu_px
        mu_y = mu_y if mu_y is not None else self.mu_y
        mu_py = mu_py if mu_py is not None else self.mu_py
        sigma_x = sigma_x if sigma_x is not None else self.sigma_x
        sigma_px = sigma_px if sigma_px is not None else self.sigma_px
        sigma_y = sigma_y if sigma_y is not None else self.sigma_y
        sigma_py = sigma_py if sigma_py is not None else self.sigma_py
        sigma_tau = sigma_tau if sigma_tau is not None else self.sigma_tau
        sigma_p = sigma_p if sigma_p is not None else self.sigma_p
        energy = energy if energy is not None else self.energy
        total_charge = total_charge if total_charge is not None else self.total_charge

        return self.__class__.from_parameters(
            mu_x=mu_x,
            mu_px=mu_px,
            mu_y=mu_y,
            mu_py=mu_py,
            sigma_x=sigma_x,
            sigma_px=sigma_px,
            sigma_y=sigma_y,
            sigma_py=sigma_py,
            sigma_tau=sigma_tau,
            sigma_p=sigma_p,
            energy=energy,
            total_charge=total_charge,
            device=device,
            dtype=dtype,
        )

    def linspaced(self, num_particles: int) -> ParticleBeam:
        """
        Create a `ParticleBeam` beam with the same parameters as this beam and
        `num_particles` particles evenly distributed in the beam.

        :param num_particles: Number of particles to create.
        :return: `ParticleBeam` with `num_particles` particles.
        """
        return ParticleBeam.make_linspaced(
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
            device=self._mu.device,
            dtype=self._mu.dtype,
        )

    @property
    def mu_x(self) -> torch.Tensor:
        return self._mu[..., 0]

    @property
    def sigma_x(self) -> torch.Tensor:
        return torch.sqrt(torch.clamp_min(self._cov[..., 0, 0], 1e-20))

    @property
    def mu_px(self) -> torch.Tensor:
        return self._mu[..., 1]

    @property
    def sigma_px(self) -> torch.Tensor:
        return torch.sqrt(torch.clamp_min(self._cov[..., 1, 1], 1e-20))

    @property
    def mu_y(self) -> torch.Tensor:
        return self._mu[..., 2]

    @property
    def sigma_y(self) -> torch.Tensor:
        return torch.sqrt(torch.clamp_min(self._cov[..., 2, 2], 1e-20))

    @property
    def mu_py(self) -> torch.Tensor:
        return self._mu[..., 3]

    @property
    def sigma_py(self) -> torch.Tensor:
        return torch.sqrt(torch.clamp_min(self._cov[..., 3, 3], 1e-20))

    @property
    def mu_tau(self) -> torch.Tensor:
        return self._mu[..., 4]

    @property
    def sigma_tau(self) -> torch.Tensor:
        return torch.sqrt(torch.clamp_min(self._cov[..., 4, 4], 1e-20))

    @property
    def mu_p(self) -> torch.Tensor:
        return self._mu[..., 5]

    @property
    def sigma_p(self) -> torch.Tensor:
        return torch.sqrt(torch.clamp_min(self._cov[..., 5, 5], 1e-20))

    @property
    def sigma_xpx(self) -> torch.Tensor:
        return self._cov[..., 0, 1]

    @property
    def sigma_ypy(self) -> torch.Tensor:
        return self._cov[..., 2, 3]

    def clone(self) -> "ParameterBeam":
        return ParameterBeam(
            mu=self._mu.clone(),
            cov=self._cov.clone(),
            energy=self.energy.clone(),
            total_charge=self.total_charge.clone(),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mu_x={repr(self.mu_x)},"
            f" mu_px={repr(self.mu_px)}, mu_y={repr(self.mu_y)},"
            f" mu_py={repr(self.mu_py)}, sigma_x={repr(self.sigma_x)},"
            f" sigma_px={repr(self.sigma_px)}, sigma_y={repr(self.sigma_y)},"
            f" sigma_py={repr(self.sigma_py)}, sigma_tau={repr(self.sigma_tau)},"
            f" sigma_p={repr(self.sigma_p)}, energy={repr(self.energy)}),"
            f" total_charge={repr(self.total_charge)})"
        )
