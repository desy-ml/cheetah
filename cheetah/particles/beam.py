from typing import Optional

import torch
from scipy.constants import physical_constants
from torch import nn

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
        sigma_s: Optional[torch.Tensor] = None,
        sigma_p: Optional[torch.Tensor] = None,
        cor_s: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        total_charge: Optional[torch.Tensor] = None,
        device=None,
        dtype=torch.float32,
    ) -> "Beam":
        """
        Create a beam from twiss parameters.

        :param beta_x: Beta function in x direction in meters.
        :param alpha_x: Alpha function in x direction in rad.
        :param emittance_x: Emittance in x direction in m*rad.
        :param beta_y: Beta function in y direction in meters.
        :param alpha_y: Alpha function in y direction in rad.
        :param emittance_y: Emittance in y direction in m*rad.
        :param sigma_s: Sigma of the particle distribution in s direction in meters.
        :param sigma_p: Sigma of the particle distribution in p direction in meters.
        :param cor_s: Correlation of the particle distribution in s direction.
        :param energy: Energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        :param device: Device to create the beam on.
        :param dtype: Data type of the beam.
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
        :param mu_xp: Center of the particle distribution on x' in rad.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_yp: Center of the particle distribution on y' in rad.
        :param sigma_x: Sigma of the particle distribution in x direction in meters.
        :param sigma_xp: Sigma of the particle distribution in x' direction in rad.
        :param sigma_y: Sigma of the particle distribution in y direction in meters.
        :param sigma_yp: Sigma of the particle distribution in y' direction in rad.
        :param sigma_s: Sigma of the particle distribution in s direction in meters.
        :param sigma_p: Sigma of the particle distribution in p direction,
        dimensionless.
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
                self.sigma_x**2 * self.sigma_xp**2 - self.sigma_xxp**2,
                torch.finfo(self.sigma_x.dtype).tiny,
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
        """Alpha function in x direction in rad."""
        return -self.sigma_xxp / self.emittance_x

    @property
    def emittance_y(self) -> torch.Tensor:
        """Emittance of the beam in y direction in m*rad."""
        return torch.sqrt(
            torch.clamp_min(
                self.sigma_y**2 * self.sigma_yp**2 - self.sigma_yyp**2,
                torch.finfo(self.sigma_y.dtype).tiny,
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
        """Alpha function in y direction in rad."""
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
