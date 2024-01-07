from typing import Optional

import numpy as np
import torch
from scipy.constants import physical_constants
from torch import nn
from torch.distributions import MultivariateNormal

electron_mass_eV = torch.tensor(
    physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
)


class Beam(nn.Module):
    empty = "I'm an empty beam!"

    @classmethod
    def from_parameters(
        cls,
        mu_x: Optional[torch.Tensor] = None,
        mu_xp: Optional[torch.Tensor] = None,
        mu_y: Optional[torch.Tensor] = None,
        mu_yp: Optional[torch.Tensor] = None,
        sigma_x: Optional[torch.Tensor] = None,
        sigma_xp: Optional[torch.Tensor] = None,
        sigma_y: Optional[torch.Tensor] = None,
        sigma_yp: Optional[torch.Tensor] = None,
        sigma_s: Optional[torch.Tensor] = None,
        sigma_p: Optional[torch.Tensor] = None,
        cor_x: Optional[torch.Tensor] = None,
        cor_y: Optional[torch.Tensor] = None,
        cor_s: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        total_charge: Optional[torch.Tensor] = None,
    ) -> "Beam":
        """
        Create beam that with given beam parameters.

        :param n: Number of particles to generate.
        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_xp: Center of the particle distribution on x'=px/px'
            (trace space) in rad.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_yp: Center of the particle distribution on y' in rad.
        :param sigma_x: Sigma of the particle distribution in x direction in meters.
        :param sigma_xp: Sigma of the particle distribution in x' direction in rad.
        :param sigma_y: Sigma of the particle distribution in y direction in meters.
        :param sigma_yp: Sigma of the particle distribution in y' direction in rad.
        :param sigma_s: Sigma of the particle distribution in s direction in meters.
        :param sigma_p: Sigma of the particle distribution in p direction in meters.
        :param energy: Energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        """
        raise NotImplementedError

    @classmethod
    def from_twiss(
        cls,
        beta_x: Optional[torch.Tensor] = None,
        alpha_x: Optional[torch.Tensor] = None,
        emittance_x: Optional[torch.Tensor] = None,
        beta_y: Optional[torch.Tensor] = None,
        alpha_y: Optional[torch.Tensor] = None,
        emittance_y: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        total_charge: Optional[torch.Tensor] = None,
    ) -> "Beam":
        """
        Create a beam from twiss parameters.

        :param beta_x: Beta function in x direction in meters.
        :param alpha_x: Alpha function in x direction in meters.
        :param emittance_x: Emittance in x direction.
        :param beta_y: Beta function in y direction in meters.
        :param alpha_y: Alpha function in y direction in meters.
        :param emittance_y: Emittance in y direction.
        :param energy: Energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        """
        raise NotImplementedError

    @classmethod
    def from_ocelot(cls, parray) -> "Beam":
        """
        Convert an Ocelot ParticleArray `parray` to a Cheetah Beam.
        """
        raise NotImplementedError

    @classmethod
    def from_astra(cls, path: str, **kwargs) -> "Beam":
        """Load an Astra particle distribution as a Cheetah Beam."""
        raise NotImplementedError

    def transformed_to(
        self,
        mu_x: Optional[torch.Tensor] = None,
        mu_xp: Optional[torch.Tensor] = None,
        mu_y: Optional[torch.Tensor] = None,
        mu_yp: Optional[torch.Tensor] = None,
        sigma_x: Optional[torch.Tensor] = None,
        sigma_xp: Optional[torch.Tensor] = None,
        sigma_y: Optional[torch.Tensor] = None,
        sigma_yp: Optional[torch.Tensor] = None,
        sigma_s: Optional[torch.Tensor] = None,
        sigma_p: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        total_charge: Optional[torch.Tensor] = None,
    ) -> "Beam":
        """
        Create version of this beam that is transformed to new beam parameters.

        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_xp: Center of the particle distribution on px in meters.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_yp: Center of the particle distribution on py in meters.
        :param sigma_x: Sigma of the particle distribution in x direction in meters.
        :param sigma_xp: Sigma of the particle distribution in px direction in meters.
        :param sigma_y: Sigma of the particle distribution in y direction in meters.
        :param sigma_yp: Sigma of the particle distribution in py direction in meters.
        :param sigma_s: Sigma of the particle distribution in s direction in meters.
        :param sigma_p: Sigma of the particle distribution in p direction in meters.
        :param energy: Energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        """
        # Figure out batch size of the original beam and check that passed arguments
        # have the same batch size
        shape = self.mu_x.shape
        not_nones = [
            argument
            for argument in [
                mu_x,
                mu_xp,
                mu_y,
                mu_yp,
                sigma_x,
                sigma_xp,
                sigma_y,
                sigma_yp,
                sigma_s,
                sigma_p,
                energy,
                total_charge,
            ]
            if argument is not None
        ]
        if len(not_nones) > 0:
            assert all(
                argument.shape == shape for argument in not_nones
            ), "Arguments must have the same shape."

        mu_x = mu_x if mu_x is not None else self.mu_x
        mu_xp = mu_xp if mu_xp is not None else self.mu_xp
        mu_y = mu_y if mu_y is not None else self.mu_y
        mu_yp = mu_yp if mu_yp is not None else self.mu_yp
        sigma_x = sigma_x if sigma_x is not None else self.sigma_x
        sigma_xp = sigma_xp if sigma_xp is not None else self.sigma_xp
        sigma_y = sigma_y if sigma_y is not None else self.sigma_y
        sigma_yp = sigma_yp if sigma_yp is not None else self.sigma_yp
        sigma_s = sigma_s if sigma_s is not None else self.sigma_s
        sigma_p = sigma_p if sigma_p is not None else self.sigma_p
        energy = energy if energy is not None else self.energy
        total_charge = total_charge if total_charge is not None else self.total_charge

        return self.__class__.from_parameters(
            mu_x=mu_x,
            mu_xp=mu_xp,
            mu_y=mu_y,
            mu_yp=mu_yp,
            sigma_x=sigma_x,
            sigma_xp=sigma_xp,
            sigma_y=sigma_y,
            sigma_yp=sigma_yp,
            sigma_s=sigma_s,
            sigma_p=sigma_p,
            energy=energy,
            total_charge=total_charge,
        )

    @property
    def parameters(self) -> dict:
        return {
            "mu_x": self.mu_x,
            "mu_xp": self.mu_xp,
            "mu_y": self.mu_y,
            "mu_yp": self.mu_yp,
            "sigma_x": self.sigma_x,
            "sigma_xp": self.sigma_xp,
            "sigma_y": self.sigma_y,
            "sigma_yp": self.sigma_yp,
            "sigma_s": self.sigma_s,
            "sigma_p": self.sigma_p,
            "energy": self.energy,
        }

    @property
    def mu_x(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def sigma_x(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def mu_xp(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def sigma_xp(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def mu_y(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def sigma_y(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def mu_yp(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def sigma_yp(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def mu_s(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def sigma_s(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def mu_p(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def sigma_p(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def relativistic_gamma(self) -> torch.Tensor:
        return self.energy / electron_mass_eV

    @property
    def relativistic_beta(self) -> torch.Tensor:
        relativistic_beta = torch.ones_like(self.relativistic_gamma)
        relativistic_beta[torch.abs(self.relativistic_gamma) > 0] = torch.sqrt(
            1 - 1 / (self.relativistic_gamma[self.relativistic_gamma > 0] ** 2)
        )
        return relativistic_beta

    @property
    def sigma_xxp(self) -> torch.Tensor:
        # the covariance of (x,x') ~ $\sigma_{xx'}$
        raise NotImplementedError

    @property
    def sigma_yyp(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def emittance_x(self) -> torch.Tensor:
        """Emittance of the beam in x direction in m*rad."""
        return torch.sqrt(
            torch.clamp_min(
                self.sigma_x**2 * self.sigma_xp**2 - self.sigma_xxp**2, 0.0
            )
        )

    @property
    def normalized_emittance_x(self) -> torch.Tensor:
        """Normalized emittance of the beam in x direction in m*rad."""
        return self.emittance_x * self.relativistic_beta * self.relativistic_gamma

    @property
    def beta_x(self) -> torch.Tensor:
        """Beta function in x direction in meters."""
        return self.sigma_x**2 / self.emittance_x

    @property
    def alpha_x(self) -> torch.Tensor:
        return -self.sigma_xxp / self.emittance_x

    @property
    def emittance_y(self) -> torch.Tensor:
        """Emittance of the beam in y direction in m*rad."""
        return torch.sqrt(
            torch.clamp_min(
                self.sigma_y**2 * self.sigma_yp**2 - self.sigma_yyp**2, 0.0
            )
        )

    @property
    def normalized_emittance_y(self) -> torch.Tensor:
        """Normalized emittance of the beam in y direction in m*rad."""
        return self.emittance_y * self.relativistic_beta * self.relativistic_gamma

    @property
    def beta_y(self) -> torch.Tensor:
        """Beta function in y direction in meters."""
        return self.sigma_y**2 / self.emittance_y

    @property
    def alpha_y(self) -> torch.Tensor:
        return -self.sigma_yyp / self.emittance_y

    def broadcast(self, shape: torch.Size) -> "Beam":
        """Broadcast beam to new shape."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mu_x={self.mu_x}, mu_xp={self.mu_xp},"
            f" mu_y={self.mu_y}, mu_yp={self.mu_yp}, sigma_x={self.sigma_x},"
            f" sigma_xp={self.sigma_xp}, sigma_y={self.sigma_y},"
            f" sigma_yp={self.sigma_yp}, sigma_s={self.sigma_s},"
            f" sigma_p={self.sigma_p}, energy={self.energy}),"
            f" total_charge={self.total_charge})"
        )


class ParameterBeam(Beam):
    """
    Beam of charged particles, where each particle is simulated.

    :param mu: Mu vector of the beam.
    :param cov: Covariance matrix of the beam.
    :param energy: Energy of the beam in eV.
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
        dtype=torch.float32,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self._mu = torch.as_tensor(mu, **factory_kwargs)
        self._cov = torch.as_tensor(cov, **factory_kwargs)
        total_charge = (
            total_charge
            if total_charge is not None
            else torch.tensor([0.0], **factory_kwargs)
        )
        self.total_charge = torch.as_tensor(total_charge, **factory_kwargs)
        self.energy = torch.as_tensor(energy, **factory_kwargs)

    @classmethod
    def from_parameters(
        cls,
        mu_x: Optional[torch.Tensor] = None,
        mu_xp: Optional[torch.Tensor] = None,
        mu_y: Optional[torch.Tensor] = None,
        mu_yp: Optional[torch.Tensor] = None,
        sigma_x: Optional[torch.Tensor] = None,
        sigma_xp: Optional[torch.Tensor] = None,
        sigma_y: Optional[torch.Tensor] = None,
        sigma_yp: Optional[torch.Tensor] = None,
        sigma_s: Optional[torch.Tensor] = None,
        sigma_p: Optional[torch.Tensor] = None,
        cor_x: Optional[torch.Tensor] = None,
        cor_y: Optional[torch.Tensor] = None,
        cor_s: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        total_charge: Optional[torch.Tensor] = None,
        device=None,
        dtype=torch.float32,
    ) -> "ParameterBeam":
        # Figure out if arguments were passed, figure out their shape
        not_nones = [
            argument
            for argument in [
                mu_x,
                mu_xp,
                mu_y,
                mu_yp,
                sigma_x,
                sigma_xp,
                sigma_y,
                sigma_yp,
                sigma_s,
                sigma_p,
                cor_x,
                cor_y,
                cor_s,
                energy,
                total_charge,
            ]
            if argument is not None
        ]
        shape = not_nones[0].shape if len(not_nones) > 0 else torch.Size([1])
        if len(not_nones) > 1:
            assert all(
                argument.shape == shape for argument in not_nones
            ), "Arguments must have the same shape."

        # Set default values without function call in function signature
        mu_x = mu_x if mu_x is not None else torch.full(shape, 0.0)
        mu_xp = mu_xp if mu_xp is not None else torch.full(shape, 0.0)
        mu_y = mu_y if mu_y is not None else torch.full(shape, 0.0)
        mu_yp = mu_yp if mu_yp is not None else torch.full(shape, 0.0)
        sigma_x = sigma_x if sigma_x is not None else torch.full(shape, 175e-9)
        sigma_xp = sigma_xp if sigma_xp is not None else torch.full(shape, 2e-7)
        sigma_y = sigma_y if sigma_y is not None else torch.full(shape, 175e-9)
        sigma_yp = sigma_yp if sigma_yp is not None else torch.full(shape, 2e-7)
        sigma_s = sigma_s if sigma_s is not None else torch.full(shape, 1e-6)
        sigma_p = sigma_p if sigma_p is not None else torch.full(shape, 1e-6)
        cor_x = cor_x if cor_x is not None else torch.full(shape, 0.0)
        cor_y = cor_y if cor_y is not None else torch.full(shape, 0.0)
        cor_s = cor_s if cor_s is not None else torch.full(shape, 0.0)
        energy = energy if energy is not None else torch.full(shape, 1e8)
        total_charge = (
            total_charge if total_charge is not None else torch.full(shape, 0.0)
        )

        mu = torch.stack(
            [
                mu_x,
                mu_xp,
                mu_y,
                mu_yp,
                torch.full(shape, 0.0),
                torch.full(shape, 0.0),
                torch.full(shape, 1.0),
            ],
            dim=1,
        )

        cov = torch.zeros(shape[0], 7, 7)
        cov[:, 0, 0] = sigma_x**2
        cov[:, 0, 1] = cor_x
        cov[:, 1, 0] = cor_x
        cov[:, 1, 1] = sigma_xp**2
        cov[:, 2, 2] = sigma_y**2
        cov[:, 2, 3] = cor_y
        cov[:, 3, 2] = cor_y
        cov[:, 3, 3] = sigma_yp**2
        cov[:, 4, 4] = sigma_s**2
        cov[:, 4, 5] = cor_s
        cov[:, 5, 4] = cor_s
        cov[:, 5, 5] = sigma_p**2

        return cls(
            mu=mu, cov=cov, energy=energy, total_charge=total_charge, device=device
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
        sigma_s: Optional[torch.Tensor] = None,
        sigma_p: Optional[torch.Tensor] = None,
        cor_s: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        total_charge: Optional[torch.Tensor] = None,
        device=None,
        dtype=torch.float32,
    ) -> "ParameterBeam":
        # Figure out if arguments were passed, figure out their shape
        not_nones = [
            argument
            for argument in [
                beta_x,
                alpha_x,
                emittance_x,
                beta_y,
                alpha_y,
                emittance_y,
                sigma_s,
                sigma_p,
                cor_s,
                energy,
                total_charge,
            ]
            if argument is not None
        ]
        shape = not_nones[0].shape if len(not_nones) > 0 else torch.Size([1])
        if len(not_nones) > 1:
            assert all(
                argument.shape == shape for argument in not_nones
            ), "Arguments must have the same shape."

        # Set default values without function call in function signature
        beta_x = beta_x if beta_x is not None else torch.full(shape, 0.0)
        alpha_x = alpha_x if alpha_x is not None else torch.full(shape, 0.0)
        emittance_x = emittance_x if emittance_x is not None else torch.full(shape, 0.0)
        beta_y = beta_y if beta_y is not None else torch.full(shape, 0.0)
        alpha_y = alpha_y if alpha_y is not None else torch.full(shape, 0.0)
        emittance_y = emittance_y if emittance_y is not None else torch.full(shape, 0.0)
        sigma_s = sigma_s if sigma_s is not None else torch.full(shape, 1e-6)
        sigma_p = sigma_p if sigma_p is not None else torch.full(shape, 1e-6)
        cor_s = cor_s if cor_s is not None else torch.full(shape, 0.0)
        energy = energy if energy is not None else torch.full(shape, 1e8)
        total_charge = (
            total_charge if total_charge is not None else torch.full(shape, 0.0)
        )

        sigma_x = torch.sqrt(emittance_x * beta_x)
        sigma_xp = torch.sqrt(emittance_x * (1 + alpha_x**2) / beta_x)
        sigma_y = torch.sqrt(emittance_y * beta_y)
        sigma_yp = torch.sqrt(emittance_y * (1 + alpha_y**2) / beta_y)
        cor_x = -emittance_x * alpha_x
        cor_y = -emittance_y * alpha_y
        return cls.from_parameters(
            sigma_x=sigma_x,
            sigma_xp=sigma_xp,
            sigma_y=sigma_y,
            sigma_yp=sigma_yp,
            sigma_s=sigma_s,
            sigma_p=sigma_p,
            energy=energy,
            cor_s=cor_s,
            cor_x=cor_x,
            cor_y=cor_y,
            total_charge=total_charge,
            device=device,
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
        from cheetah.converters.astralavista import from_astrabeam

        particles, energy, particle_charges = from_astrabeam(path)
        mu = torch.ones(7)
        mu[:6] = torch.tensor(particles.mean(axis=0))

        cov = torch.zeros(7, 7)
        cov[:6, :6] = torch.tensor(np.cov(particles.transpose()), dtype=torch.float32)

        total_charge = torch.tensor(np.sum(particle_charges), dtype=torch.float32)

        return cls(
            mu=mu.unsqueeze(0),
            cov=cov.unsqueeze(0),
            energy=torch.tensor(energy, dtype=torch.float32).unsqueeze(0),
            total_charge=total_charge.unsqueeze(0),
            device=device,
            dtype=dtype,
        )

    def transformed_to(
        self,
        mu_x: Optional[torch.Tensor] = None,
        mu_xp: Optional[torch.Tensor] = None,
        mu_y: Optional[torch.Tensor] = None,
        mu_yp: Optional[torch.Tensor] = None,
        sigma_x: Optional[torch.Tensor] = None,
        sigma_xp: Optional[torch.Tensor] = None,
        sigma_y: Optional[torch.Tensor] = None,
        sigma_yp: Optional[torch.Tensor] = None,
        sigma_s: Optional[torch.Tensor] = None,
        sigma_p: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        total_charge: Optional[torch.Tensor] = None,
        device=None,
        dtype=torch.float32,
    ) -> "ParameterBeam":
        """
        Create version of this beam that is transformed to new beam parameters.

        :param n: Number of particles to generate.
        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_xp: Center of the particle distribution on px in meters.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_yp: Center of the particle distribution on py in meters.
        :param sigma_x: Sigma of the particle distribution in x direction in meters.
        :param sigma_xp: Sigma of the particle distribution in px direction in meters.
        :param sigma_y: Sigma of the particle distribution in y direction in meters.
        :param sigma_yp: Sigma of the particle distribution in py direction in meters.
        :param sigma_s: Sigma of the particle distribution in s direction in meters.
        :param sigma_p: Sigma of the particle distribution in p direction in meters.
        :param energy: Energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        """
        device = device if device is not None else self.mu_x.device
        dtype = dtype if dtype is not None else self.mu_x.dtype

        # Figure out batch size of the original beam and check that passed arguments
        # have the same batch size
        shape = self.mu_x.shape
        not_nones = [
            argument
            for argument in [
                mu_x,
                mu_xp,
                mu_y,
                mu_yp,
                sigma_x,
                sigma_xp,
                sigma_y,
                sigma_yp,
                sigma_s,
                sigma_p,
                energy,
                total_charge,
            ]
            if argument is not None
        ]
        if len(not_nones) > 0:
            assert all(
                argument.shape == shape for argument in not_nones
            ), "Arguments must have the same shape."

        mu_x = mu_x if mu_x is not None else self.mu_x
        mu_xp = mu_xp if mu_xp is not None else self.mu_xp
        mu_y = mu_y if mu_y is not None else self.mu_y
        mu_yp = mu_yp if mu_yp is not None else self.mu_yp
        sigma_x = sigma_x if sigma_x is not None else self.sigma_x
        sigma_xp = sigma_xp if sigma_xp is not None else self.sigma_xp
        sigma_y = sigma_y if sigma_y is not None else self.sigma_y
        sigma_yp = sigma_yp if sigma_yp is not None else self.sigma_yp
        sigma_s = sigma_s if sigma_s is not None else self.sigma_s
        sigma_p = sigma_p if sigma_p is not None else self.sigma_p
        energy = energy if energy is not None else self.energy
        total_charge = total_charge if total_charge is not None else self.total_charge

        return self.__class__.from_parameters(
            mu_x=mu_x,
            mu_xp=mu_xp,
            mu_y=mu_y,
            mu_yp=mu_yp,
            sigma_x=sigma_x,
            sigma_xp=sigma_xp,
            sigma_y=sigma_y,
            sigma_yp=sigma_yp,
            sigma_s=sigma_s,
            sigma_p=sigma_p,
            energy=energy,
            total_charge=total_charge,
            device=device,
            dtype=dtype,
        )

    @property
    def mu_x(self) -> torch.Tensor:
        return self._mu[:, 0]

    @property
    def sigma_x(self) -> torch.Tensor:
        return torch.sqrt(self._cov[:, 0, 0])

    @property
    def mu_xp(self) -> torch.Tensor:
        return self._mu[:, 1]

    @property
    def sigma_xp(self) -> torch.Tensor:
        return torch.sqrt(self._cov[:, 1, 1])

    @property
    def mu_y(self) -> torch.Tensor:
        return self._mu[:, 2]

    @property
    def sigma_y(self) -> torch.Tensor:
        return torch.sqrt(self._cov[:, 2, 2])

    @property
    def mu_yp(self) -> torch.Tensor:
        return self._mu[:, 3]

    @property
    def sigma_yp(self) -> torch.Tensor:
        return torch.sqrt(self._cov[:, 3, 3])

    @property
    def mu_s(self) -> torch.Tensor:
        return self._mu[:, 4]

    @property
    def sigma_s(self) -> torch.Tensor:
        return torch.sqrt(self._cov[:, 4, 4])

    @property
    def mu_p(self) -> torch.Tensor:
        return self._mu[:, 5]

    @property
    def sigma_p(self) -> torch.Tensor:
        return torch.sqrt(self._cov[:, 5, 5])

    @property
    def sigma_xxp(self) -> torch.Tensor:
        return self._cov[:, 0, 1]

    @property
    def sigma_yyp(self) -> torch.Tensor:
        return self._cov[:, 2, 3]

    def broadcast(self, shape: torch.Size) -> "ParameterBeam":
        return self.__class__(
            mu=self._mu.repeat((*shape, 1)),
            cov=self._cov.repeat((*shape, 1, 1)),
            energy=self.energy.repeat(shape),
            total_charge=self.total_charge.repeat(shape),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mu_x={repr(self.mu_x)},"
            f" mu_xp={repr(self.mu_xp)}, mu_y={repr(self.mu_y)},"
            f" mu_yp={repr(self.mu_yp)}, sigma_x={repr(self.sigma_x)},"
            f" sigma_xp={repr(self.sigma_xp)}, sigma_y={repr(self.sigma_y)},"
            f" sigma_yp={repr(self.sigma_yp)}, sigma_s={repr(self.sigma_s)},"
            f" sigma_p={repr(self.sigma_p)}, energy={repr(self.energy)}),"
            f" total_charge={repr(self.total_charge)})"
        )


class ParticleBeam(Beam):
    """
    Beam of charged particles, where each particle is simulated.

    :param particles: List of 7-dimensional particle vectors.
    :param energy: Energy of the beam in eV.
    :param total_charge: Total charge of the beam in C.
    :param device: Device to move the beam's particle array to. If set to `"auto"` a
        CUDA GPU is selected if available. The CPU is used otherwise.
    """

    def __init__(
        self,
        particles: torch.Tensor,
        energy: torch.Tensor,
        particle_charges: Optional[torch.Tensor] = None,
        device=None,
        dtype=torch.float32,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        assert (
            particles.shape[-2] > 0 and particles.shape[-1] == 7
        ), "Particle vectors must be 7-dimensional."

        self.particles = particles.to(**factory_kwargs)
        self.particle_charges = (
            particle_charges.to(**factory_kwargs)
            if particle_charges is not None
            else torch.zeros(particles.shape[:2], **factory_kwargs)
        )
        self.energy = energy.to(**factory_kwargs)

    @classmethod
    def from_parameters(
        cls,
        num_particles: Optional[torch.Tensor] = None,
        mu_x: Optional[torch.Tensor] = None,
        mu_y: Optional[torch.Tensor] = None,
        mu_xp: Optional[torch.Tensor] = None,
        mu_yp: Optional[torch.Tensor] = None,
        sigma_x: Optional[torch.Tensor] = None,
        sigma_y: Optional[torch.Tensor] = None,
        sigma_xp: Optional[torch.Tensor] = None,
        sigma_yp: Optional[torch.Tensor] = None,
        sigma_s: Optional[torch.Tensor] = None,
        sigma_p: Optional[torch.Tensor] = None,
        cor_x: Optional[torch.Tensor] = None,
        cor_y: Optional[torch.Tensor] = None,
        cor_s: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        total_charge: Optional[torch.Tensor] = None,
        device=None,
        dtype=torch.float32,
    ) -> "ParticleBeam":
        """
        Generate Cheetah Beam of random particles.

        :param num_particles: Number of particles to generate.
        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_xp: Center of the particle distribution on px in meters.
        :param mu_yp: Center of the particle distribution on py in meters.
        :param sigma_x: Sigma of the particle distribution in x direction in meters.
        :param sigma_y: Sigma of the particle distribution in y direction in meters.
        :param sigma_xp: Sigma of the particle distribution in px direction in meters.
        :param sigma_yp: Sigma of the particle distribution in py direction in meters.
        :param sigma_s: Sigma of the particle distribution in s direction in meters.
        :param sigma_p: Sigma of the particle distribution in p direction in meters.
        :param cor_x: Correlation between x and xp.
        :param cor_y: Correlation between y and yp.
        :param cor_s: Correlation between s and p.
        :param energy: Energy of the beam in eV.
        :total_charge: Total charge of the beam in C.
        :param device: Device to move the beam's particle array to. If set to `"auto"` a
            CUDA GPU is selected if available. The CPU is used otherwise.
        """
        # Figure out if arguments were passed, figure out their shape
        not_nones = [
            argument
            for argument in [
                mu_x,
                mu_xp,
                mu_y,
                mu_yp,
                sigma_x,
                sigma_xp,
                sigma_y,
                sigma_yp,
                sigma_s,
                sigma_p,
                cor_x,
                cor_y,
                cor_s,
                energy,
                total_charge,
            ]
            if argument is not None
        ]
        shape = not_nones[0].shape if len(not_nones) > 0 else torch.Size([1])
        if len(not_nones) > 1:
            assert all(
                argument.shape == shape for argument in not_nones
            ), "Arguments must have the same shape."

        # Set default values without function call in function signature
        num_particles = (
            num_particles if num_particles is not None else torch.tensor(100_000)
        )
        mu_x = mu_x if mu_x is not None else torch.full(shape, 0.0)
        mu_xp = mu_xp if mu_xp is not None else torch.full(shape, 0.0)
        mu_y = mu_y if mu_y is not None else torch.full(shape, 0.0)
        mu_yp = mu_yp if mu_yp is not None else torch.full(shape, 0.0)
        sigma_x = sigma_x if sigma_x is not None else torch.full(shape, 175e-9)
        sigma_xp = sigma_xp if sigma_xp is not None else torch.full(shape, 2e-7)
        sigma_y = sigma_y if sigma_y is not None else torch.full(shape, 175e-9)
        sigma_yp = sigma_yp if sigma_yp is not None else torch.full(shape, 2e-7)
        sigma_s = sigma_s if sigma_s is not None else torch.full(shape, 1e-6)
        sigma_p = sigma_p if sigma_p is not None else torch.full(shape, 1e-6)
        cor_x = cor_x if cor_x is not None else torch.full(shape, 0.0)
        cor_y = cor_y if cor_y is not None else torch.full(shape, 0.0)
        cor_s = cor_s if cor_s is not None else torch.full(shape, 0.0)
        energy = energy if energy is not None else torch.full(shape, 1e8)
        total_charge = (
            total_charge if total_charge is not None else torch.full(shape, 0.0)
        )
        particle_charges = (
            torch.ones((shape[0], num_particles), device=device, dtype=dtype)
            * total_charge.view(-1, 1)
            / num_particles
        )

        mean = torch.stack(
            [mu_x, mu_xp, mu_y, mu_yp, torch.full(shape, 0.0), torch.full(shape, 0.0)],
            dim=1,
        )

        cov = torch.zeros(shape[0], 6, 6)
        cov[:, 0, 0] = sigma_x**2
        cov[:, 0, 1] = cor_x
        cov[:, 1, 0] = cor_x
        cov[:, 1, 1] = sigma_xp**2
        cov[:, 2, 2] = sigma_y**2
        cov[:, 2, 3] = cor_y
        cov[:, 3, 2] = cor_y
        cov[:, 3, 3] = sigma_yp**2
        cov[:, 4, 4] = sigma_s**2
        cov[:, 4, 5] = cor_s
        cov[:, 5, 4] = cor_s
        cov[:, 5, 5] = sigma_p**2

        particles = torch.ones((shape[0], num_particles, 7))
        distributions = [
            MultivariateNormal(sample_mean, covariance_matrix=sample_cov)
            for sample_mean, sample_cov in zip(mean, cov)
        ]
        particles[:, :, :6] = torch.stack(
            [distribution.sample((num_particles,)) for distribution in distributions],
            dim=0,
        )

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
        num_particles: Optional[torch.Tensor] = None,
        beta_x: Optional[torch.Tensor] = None,
        alpha_x: Optional[torch.Tensor] = None,
        emittance_x: Optional[torch.Tensor] = None,
        beta_y: Optional[torch.Tensor] = None,
        alpha_y: Optional[torch.Tensor] = None,
        emittance_y: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        sigma_s: Optional[torch.Tensor] = None,
        sigma_p: Optional[torch.Tensor] = None,
        cor_s: Optional[torch.Tensor] = None,
        total_charge: Optional[torch.Tensor] = None,
        device=None,
        dtype=torch.float32,
    ) -> "ParticleBeam":
        # Figure out if arguments were passed, figure out their shape
        not_nones = [
            argument
            for argument in [
                beta_x,
                alpha_x,
                emittance_x,
                beta_y,
                alpha_y,
                emittance_y,
                energy,
                sigma_s,
                sigma_p,
                cor_s,
                total_charge,
            ]
            if argument is not None
        ]
        shape = not_nones[0].shape if len(not_nones) > 0 else torch.Size([1])
        if len(not_nones) > 1:
            assert all(
                argument.shape == shape for argument in not_nones
            ), "Arguments must have the same shape."

        # Set default values without function call in function signature
        num_particles = (
            num_particles if num_particles is not None else torch.tensor(1_000_000)
        )
        beta_x = beta_x if beta_x is not None else torch.full(shape, 0.0)
        alpha_x = alpha_x if alpha_x is not None else torch.full(shape, 0.0)
        emittance_x = emittance_x if emittance_x is not None else torch.full(shape, 0.0)
        beta_y = beta_y if beta_y is not None else torch.full(shape, 0.0)
        alpha_y = alpha_y if alpha_y is not None else torch.full(shape, 0.0)
        emittance_y = emittance_y if emittance_y is not None else torch.full(shape, 0.0)
        energy = energy if energy is not None else torch.full(shape, 1e8)
        sigma_s = sigma_s if sigma_s is not None else torch.full(shape, 1e-6)
        sigma_p = sigma_p if sigma_p is not None else torch.full(shape, 1e-6)
        cor_s = cor_s if cor_s is not None else torch.full(shape, 0.0)
        total_charge = (
            total_charge if total_charge is not None else torch.full(shape, 0.0)
        )

        sigma_x = torch.sqrt(beta_x * emittance_x)
        sigma_xp = torch.sqrt(emittance_x * (1 + alpha_x**2) / beta_x)
        sigma_y = torch.sqrt(beta_y * emittance_y)
        sigma_yp = torch.sqrt(emittance_y * (1 + alpha_y**2) / beta_y)
        cor_x = -emittance_x * alpha_x
        cor_y = -emittance_y * alpha_y

        return cls.from_parameters(
            num_particles=num_particles,
            mu_x=torch.full(shape, 0.0),
            mu_xp=torch.full(shape, 0.0),
            mu_y=torch.full(shape, 0.0),
            mu_yp=torch.full(shape, 0.0),
            sigma_x=sigma_x,
            sigma_xp=sigma_xp,
            sigma_y=sigma_y,
            sigma_yp=sigma_yp,
            sigma_s=sigma_s,
            sigma_p=sigma_p,
            energy=energy,
            cor_s=cor_s,
            cor_x=cor_x,
            cor_y=cor_y,
            total_charge=total_charge,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def make_linspaced(
        cls,
        num_particles: Optional[torch.Tensor] = None,
        mu_x: Optional[torch.Tensor] = None,
        mu_y: Optional[torch.Tensor] = None,
        mu_xp: Optional[torch.Tensor] = None,
        mu_yp: Optional[torch.Tensor] = None,
        sigma_x: Optional[torch.Tensor] = None,
        sigma_y: Optional[torch.Tensor] = None,
        sigma_xp: Optional[torch.Tensor] = None,
        sigma_yp: Optional[torch.Tensor] = None,
        sigma_s: Optional[torch.Tensor] = None,
        sigma_p: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        total_charge: Optional[torch.Tensor] = None,
        device=None,
        dtype=torch.float32,
    ) -> "ParticleBeam":
        """
        Generate Cheetah Beam of *n* linspaced particles.

        :param n: Number of particles to generate.
        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_xp: Center of the particle distribution on px in meters.
        :param mu_yp: Center of the particle distribution on py in meters.
        :param sigma_x: Sigma of the particle distribution in x direction in meters.
        :param sigma_y: Sigma of the particle distribution in y direction in meters.
        :param sigma_xp: Sigma of the particle distribution in px direction in meters.
        :param sigma_yp: Sigma of the particle distribution in py direction in meters.
        :param sigma_s: Sigma of the particle distribution in s direction in meters.
        :param sigma_p: Sigma of the particle distribution in p direction in meters.
        :param energy: Energy of the beam in eV.
        :param device: Device to move the beam's particle array to. If set to `"auto"` a
            CUDA GPU is selected if available. The CPU is used otherwise.
        """
        # Figure out if arguments were passed, figure out their shape
        not_nones = [
            argument
            for argument in [
                mu_x,
                mu_xp,
                mu_y,
                mu_yp,
                sigma_x,
                sigma_xp,
                sigma_y,
                sigma_yp,
                sigma_s,
                sigma_p,
                energy,
                total_charge,
            ]
            if argument is not None
        ]
        shape = not_nones[0].shape if len(not_nones) > 0 else torch.Size([1])
        if len(not_nones) > 1:
            assert all(
                argument.shape == shape for argument in not_nones
            ), "Arguments must have the same shape."

        # Set default values without function call in function signature
        num_particles = num_particles if num_particles is not None else torch.tensor(10)
        mu_x = mu_x if mu_x is not None else torch.full(shape, 0.0)
        mu_xp = mu_xp if mu_xp is not None else torch.full(shape, 0.0)
        mu_y = mu_y if mu_y is not None else torch.full(shape, 0.0)
        mu_yp = mu_yp if mu_yp is not None else torch.full(shape, 0.0)
        sigma_x = sigma_x if sigma_x is not None else torch.full(shape, 175e-9)
        sigma_xp = sigma_xp if sigma_xp is not None else torch.full(shape, 2e-7)
        sigma_y = sigma_y if sigma_y is not None else torch.full(shape, 175e-9)
        sigma_yp = sigma_yp if sigma_yp is not None else torch.full(shape, 2e-7)
        sigma_s = sigma_s if sigma_s is not None else torch.full(shape, 0.0)
        sigma_p = sigma_p if sigma_p is not None else torch.full(shape, 0.0)
        energy = energy if energy is not None else torch.full(shape, 1e8)
        total_charge = (
            total_charge if total_charge is not None else torch.full(shape, 0.0)
        )

        particle_charges = (
            torch.ones((shape[0], num_particles), device=device, dtype=dtype)
            * total_charge.view(-1, 1)
            / num_particles
        )

        particles = torch.ones((shape[0], num_particles, 7))

        particles[:, :, 0] = torch.stack(
            [
                torch.linspace(
                    sample_mu_x - sample_sigma_x,
                    sample_mu_x + sample_sigma_x,
                    num_particles,
                )
                for sample_mu_x, sample_sigma_x in zip(mu_x, sigma_x)
            ],
            dim=0,
        )
        particles[:, :, 1] = torch.stack(
            [
                torch.linspace(
                    sample_mu_xp - sample_sigma_xp,
                    sample_mu_xp + sample_sigma_xp,
                    num_particles,
                )
                for sample_mu_xp, sample_sigma_xp in zip(mu_xp, sigma_xp)
            ],
            dim=0,
        )
        particles[:, :, 2] = torch.stack(
            [
                torch.linspace(
                    sample_mu_y - sample_sigma_y,
                    sample_mu_y + sample_sigma_y,
                    num_particles,
                )
                for sample_mu_y, sample_sigma_y in zip(mu_y, sigma_y)
            ],
            dim=0,
        )
        particles[:, :, 3] = torch.stack(
            [
                torch.linspace(
                    sample_mu_yp - sample_sigma_yp,
                    sample_mu_yp + sample_sigma_yp,
                    num_particles,
                )
                for sample_mu_yp, sample_sigma_yp in zip(mu_yp, sigma_yp)
            ],
            dim=0,
        )
        particles[:, :, 4] = torch.stack(
            [
                torch.linspace(
                    -sample_sigma_s, sample_sigma_s, num_particles, device=device
                )
                for sample_sigma_s in sigma_s
            ],
            dim=0,
        )
        particles[:, :, 5] = torch.stack(
            [
                torch.linspace(
                    -sample_sigma_p, sample_sigma_p, num_particles, device=device
                )
                for sample_sigma_p in sigma_p
            ],
            dim=0,
        )

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
        from cheetah.converters.astralavista import from_astrabeam

        particles, energy, particle_charges = from_astrabeam(path)
        particles_7d = torch.ones((particles.shape[0], 7))
        particles_7d[:, :6] = torch.from_numpy(particles)
        particle_charges = torch.from_numpy(particle_charges)
        return cls(
            particles=particles_7d.unsqueeze(0),
            energy=torch.tensor(energy).unsqueeze(0),
            particle_charges=particle_charges.unsqueeze(0),
            device=device,
            dtype=dtype,
        )

    def transformed_to(
        self,
        mu_x: Optional[torch.Tensor] = None,
        mu_y: Optional[torch.Tensor] = None,
        mu_xp: Optional[torch.Tensor] = None,
        mu_yp: Optional[torch.Tensor] = None,
        sigma_x: Optional[torch.Tensor] = None,
        sigma_y: Optional[torch.Tensor] = None,
        sigma_xp: Optional[torch.Tensor] = None,
        sigma_yp: Optional[torch.Tensor] = None,
        sigma_s: Optional[torch.Tensor] = None,
        sigma_p: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        total_charge: Optional[torch.Tensor] = None,
        device=None,
        dtype=torch.float32,
    ) -> "ParticleBeam":
        """
        Create version of this beam that is transformed to new beam parameters.

        :param n: Number of particles to generate.
        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_xp: Center of the particle distribution on px in meters.
        :param mu_yp: Center of the particle distribution on py in meters.
        :param sigma_x: Sigma of the particle distribution in x direction in meters.
        :param sigma_y: Sigma of the particle distribution in y direction in meters.
        :param sigma_xp: Sigma of the particle distribution in px direction in meters.
        :param sigma_yp: Sigma of the particle distribution in py direction in meters.
        :param sigma_s: Sigma of the particle distribution in s direction in meters.
        :param sigma_p: Sigma of the particle distribution in p direction in meters.
        :param energy: Energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        :param device: Device to move the beam's particle array to. If set to `"auto"` a
            CUDA GPU is selected if available. The CPU is used otherwise.
        """
        device = device if device is not None else self.mu_x.device
        dtype = dtype if dtype is not None else self.mu_x.dtype

        # Figure out batch size of the original beam and check that passed arguments
        # have the same batch size
        shape = self.mu_x.shape
        not_nones = [
            argument
            for argument in [
                mu_x,
                mu_xp,
                mu_y,
                mu_yp,
                sigma_x,
                sigma_xp,
                sigma_y,
                sigma_yp,
                sigma_s,
                sigma_p,
                energy,
                total_charge,
            ]
            if argument is not None
        ]
        if len(not_nones) > 0:
            assert all(
                argument.shape == shape for argument in not_nones
            ), "Arguments must have the same shape."

        mu_x = mu_x if mu_x is not None else self.mu_x
        mu_y = mu_y if mu_y is not None else self.mu_y
        mu_xp = mu_xp if mu_xp is not None else self.mu_xp
        mu_yp = mu_yp if mu_yp is not None else self.mu_yp
        sigma_x = sigma_x if sigma_x is not None else self.sigma_x
        sigma_y = sigma_y if sigma_y is not None else self.sigma_y
        sigma_xp = sigma_xp if sigma_xp is not None else self.sigma_xp
        sigma_yp = sigma_yp if sigma_yp is not None else self.sigma_yp
        sigma_s = sigma_s if sigma_s is not None else self.sigma_s
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
                * total_charge.view(-1, 1)
                / self.particle_charges.shape[-1]
            )

        new_mu = torch.stack(
            [mu_x, mu_xp, mu_y, mu_yp, torch.full(shape, 0.0), torch.full(shape, 0.0)],
            dim=1,
        )
        new_sigma = torch.stack(
            [sigma_x, sigma_xp, sigma_y, sigma_yp, sigma_s, sigma_p], dim=1
        )

        old_mu = torch.stack(
            [
                self.mu_x,
                self.mu_xp,
                self.mu_y,
                self.mu_yp,
                torch.full(shape, 0.0),
                torch.full(shape, 0.0),
            ],
            dim=1,
        )
        old_sigma = torch.stack(
            [
                self.sigma_x,
                self.sigma_xp,
                self.sigma_y,
                self.sigma_yp,
                self.sigma_s,
                self.sigma_p,
            ],
            dim=1,
        )

        phase_space = self.particles[:, :, :6]
        phase_space = (phase_space - old_mu.unsqueeze(1)) / old_sigma.unsqueeze(
            1
        ) * new_sigma.unsqueeze(1) + new_mu.unsqueeze(1)

        particles = torch.ones_like(self.particles)
        particles[:, :, :6] = phase_space

        return self.__class__(
            particles=particles,
            energy=energy,
            particle_charges=particle_charges,
            device=device,
            dtype=dtype,
        )

    def __len__(self) -> int:
        return int(self.num_particles)

    @property
    def total_charge(self) -> torch.Tensor:
        return torch.sum(self.particle_charges, dim=1)

    @property
    def num_particles(self) -> int:
        return self.particles.shape[1]

    @property
    def xs(self) -> Optional[torch.Tensor]:
        return self.particles[:, :, 0] if self is not Beam.empty else None

    @xs.setter
    def xs(self, value: torch.Tensor) -> None:
        self.particles[:, :, 0] = value

    @property
    def mu_x(self) -> Optional[torch.Tensor]:
        return self.xs.mean(dim=1) if self is not Beam.empty else None

    @property
    def sigma_x(self) -> Optional[torch.Tensor]:
        return self.xs.std(dim=1) if self is not Beam.empty else None

    @property
    def xps(self) -> Optional[torch.Tensor]:
        return self.particles[:, :, 1] if self is not Beam.empty else None

    @xps.setter
    def xps(self, value: torch.Tensor) -> None:
        self.particles[:, :, 1] = value

    @property
    def mu_xp(self) -> Optional[torch.Tensor]:
        return self.xps.mean(dim=1) if self is not Beam.empty else None

    @property
    def sigma_xp(self) -> Optional[torch.Tensor]:
        return self.xps.std(dim=1) if self is not Beam.empty else None

    @property
    def ys(self) -> Optional[torch.Tensor]:
        return self.particles[:, :, 2] if self is not Beam.empty else None

    @ys.setter
    def ys(self, value: torch.Tensor) -> None:
        self.particles[:, :, 2] = value

    @property
    def mu_y(self) -> Optional[float]:
        return self.ys.mean(dim=1) if self is not Beam.empty else None

    @property
    def sigma_y(self) -> Optional[torch.Tensor]:
        return self.ys.std(dim=1) if self is not Beam.empty else None

    @property
    def yps(self) -> Optional[torch.Tensor]:
        return self.particles[:, :, 3] if self is not Beam.empty else None

    @yps.setter
    def yps(self, value: torch.Tensor) -> None:
        self.particles[:, :, 3] = value

    @property
    def mu_yp(self) -> Optional[torch.Tensor]:
        return self.yps.mean(dim=1) if self is not Beam.empty else None

    @property
    def sigma_yp(self) -> Optional[torch.Tensor]:
        return self.yps.std(dim=1) if self is not Beam.empty else None

    @property
    def ss(self) -> Optional[torch.Tensor]:
        return self.particles[:, :, 4] if self is not Beam.empty else None

    @ss.setter
    def ss(self, value: torch.Tensor) -> None:
        self.particles[:, :, 4] = value

    @property
    def mu_s(self) -> Optional[torch.Tensor]:
        return self.ss.mean(dim=1) if self is not Beam.empty else None

    @property
    def sigma_s(self) -> Optional[torch.Tensor]:
        return self.ss.std(dim=1) if self is not Beam.empty else None

    @property
    def ps(self) -> Optional[torch.Tensor]:
        return self.particles[:, :, 5] if self is not Beam.empty else None

    @ps.setter
    def ps(self, value: torch.Tensor) -> None:
        self.particles[:, :, 5] = value

    @property
    def mu_p(self) -> Optional[torch.Tensor]:
        return self.ps.mean(dim=1) if self is not Beam.empty else None

    @property
    def sigma_p(self) -> Optional[torch.Tensor]:
        return self.ps.std(dim=1) if self is not Beam.empty else None

    @property
    def sigma_xxp(self) -> torch.Tensor:
        return torch.mean(
            (self.xs - self.mu_x.view(-1, 1)) * (self.xps - self.mu_xp.view(-1, 1)),
            dim=1,
        )

    @property
    def sigma_yyp(self) -> torch.Tensor:
        return torch.mean(
            (self.ys - self.mu_y.view(-1, 1)) * (self.yps - self.mu_yp.view(-1, 1)),
            dim=1,
        )

    def broadcast(self, shape: torch.Size) -> "ParticleBeam":
        return self.__class__(
            particles=self.particles.repeat((*shape, 1, 1)),
            energy=self.energy.repeat(shape),
            particle_charges=self.particle_charges.repeat((*shape, 1)),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n={repr(self.num_particles)},"
            f" mu_x={repr(self.mu_x)}, mu_xp={repr(self.mu_xp)},"
            f" mu_y={repr(self.mu_y)}, mu_yp={repr(self.mu_yp)},"
            f" sigma_x={repr(self.sigma_x)}, sigma_xp={repr(self.sigma_xp)},"
            f" sigma_y={repr(self.sigma_y)}, sigma_yp={repr(self.sigma_yp)},"
            f" sigma_s={repr(self.sigma_s)}, sigma_p={repr(self.sigma_p)},"
            f" energy={repr(self.energy)})"
            f" total_charge={repr(self.total_charge)})"
        )
