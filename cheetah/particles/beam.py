from typing import Optional

import torch
from scipy.constants import physical_constants
from torch import nn

electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


class Beam(nn.Module):
    r"""
    Parent class to represent a beam of particles. You should not instantiate this
    class directly, but use one of the subclasses.

    Cheetah uses a 7D vector to describe the state of a particle.
    It contains the 6D phase space vector (x, px, y, yp, tau, p) and an additional
    dimension (always 1) for convenient calculations.

    The phase space vectors contain the canonical variables:
    - x: Position in x direction in meters.
    - px: Horizontal momentum normalized over the reference momentum (dimensionless).
        :math:`px = \frac{P_x}{P_0}`
    - y: Position in y direction in meters.
    - py: Vertical momentum normalized over the reference momentum (dimensionless).
        :math:`py = \frac{P_y}{P_0}`
    - tau: Position in longitudinal direction in meters, relative to the reference
        particle. :math:`\tau = ct - \frac{s}{\beta_0}`, where s is the position along
        the beamline. In this notation, particle ahead of the reference particle will
        have negative :math:`\tau`.
    - p: Relative energy deviation from the reference particle (dimensionless).
        :math:`p = \frac{\Delta E}{p_0 C}`, where :math:`p_0` is the reference momentum.
        :math:`\Delta E = E - E_0`
    """

    empty = "I'm an empty beam!"

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
    ) -> "Beam":
        """
        Create beam that with given beam parameters.

        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_px: Center of the particle distribution on px, dimensionless.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_py: Center of the particle distribution on yp, dimensionless.
        :param sigma_x: Sigma of the particle distribution in x direction in meters.
        :param sigma_px: Sigma of the particle distribution in px direction,
            dimensionless.
        :param sigma_y: Sigma of the particle distribution in y direction in meters.
        :param sigma_py: Sigma of the particle distribution in yp direction,
            dimensionless.
        :param sigma_tau: Sigma of the particle distribution in longitudinal direction,
            in meters.
        :param sigma_p: Sigma of the particle distribution in p direction,
            dimensionless.
        :param cor_x: Correlation between x and px.
        :param cor_y: Correlation between y and yp.
        :param cor_tau: Correlation between tau and p.
        :param energy: Reference energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        :param device: Device to create the beam on.
        :param dtype: Data type of the beam.
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
        dtype=None,
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
    def from_ocelot(cls, parray, device=None, dtype=None) -> "Beam":
        """
        Convert an Ocelot ParticleArray `parray` to a Cheetah Beam.
        """
        raise NotImplementedError

    @classmethod
    def from_astra(cls, path: str, device=None, dtype=None) -> "Beam":
        """Load an Astra particle distribution as a Cheetah Beam."""
        raise NotImplementedError

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
    ) -> "Beam":
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
        :param sigma_py: Sigma of the particle distribution in yp direction,
            dimensionless.
        :param sigma_tau: Sigma of the particle distribution in longitudinal direction,
            in meters.
        :param sigma_p: Sigma of the particle distribution in p direction,
            dimensionless.
        :param energy: Reference energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        :param device: Device to create the beam on.
        :param dtype: Data type of the beam.
        """
        device = device if device is not None else self.mu_x.device
        dtype = dtype if dtype is not None else self.mu_x.dtype

        # Figure out vector dimensions of the original beam and check that passed
        # arguments have the same vector dimensions.
        shape = self.mu_x.shape
        not_nones = [
            argument
            for argument in [
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
            ]
            if argument is not None
        ]
        if len(not_nones) > 0:
            assert all(
                argument.shape == shape for argument in not_nones
            ), "Arguments must have the same shape."

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

    @property
    def parameters(self) -> dict:
        return {
            "mu_x": self.mu_x,
            "mu_px": self.mu_px,
            "mu_y": self.mu_y,
            "mu_py": self.mu_py,
            "sigma_x": self.sigma_x,
            "sigma_px": self.sigma_px,
            "sigma_y": self.sigma_y,
            "sigma_py": self.sigma_py,
            "sigma_tau": self.sigma_tau,
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
    def mu_px(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def sigma_px(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def mu_y(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def sigma_y(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def mu_py(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def sigma_py(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def mu_s(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def sigma_tau(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def mu_p(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def sigma_p(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def relativistic_gamma(self) -> torch.Tensor:
        """Reference relativistic gamma of the beam."""
        return self.energy / electron_mass_eV

    @property
    def relativistic_beta(self) -> torch.Tensor:
        """Reference relativistic beta of the beam."""
        relativistic_beta = torch.ones_like(self.relativistic_gamma)
        relativistic_beta[torch.abs(self.relativistic_gamma) > 0] = torch.sqrt(
            1 - 1 / (self.relativistic_gamma[self.relativistic_gamma > 0] ** 2)
        )
        return relativistic_beta

    @property
    def p0c(self) -> torch.Tensor:
        """Get the reference momentum * speed of light in eV."""
        return self.relativistic_beta * self.relativistic_gamma * electron_mass_eV

    @property
    def sigma_xpx(self) -> torch.Tensor:
        # The covariance of (x,px) ~ $\sigma_{xpx}$
        raise NotImplementedError

    @property
    def sigma_ypy(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def emittance_x(self) -> torch.Tensor:
        """Emittance of the beam in x direction in m."""
        return torch.sqrt(
            torch.clamp_min(
                self.sigma_x**2 * self.sigma_px**2 - self.sigma_xpx**2,
                torch.finfo(self.sigma_x.dtype).tiny,
            )
        )

    @property
    def normalized_emittance_x(self) -> torch.Tensor:
        """Normalized emittance of the beam in x direction in m."""
        return self.emittance_x * self.relativistic_beta * self.relativistic_gamma

    @property
    def beta_x(self) -> torch.Tensor:
        """Beta function in x direction in meters."""
        return self.sigma_x**2 / self.emittance_x

    @property
    def alpha_x(self) -> torch.Tensor:
        """Alpha function in x direction, dimensionless."""
        return -self.sigma_xpx / self.emittance_x

    @property
    def emittance_y(self) -> torch.Tensor:
        """Emittance of the beam in y direction in m."""
        return torch.sqrt(
            torch.clamp_min(
                self.sigma_y**2 * self.sigma_py**2 - self.sigma_ypy**2,
                torch.finfo(self.sigma_y.dtype).tiny,
            )
        )

    @property
    def normalized_emittance_y(self) -> torch.Tensor:
        """Normalized emittance of the beam in y direction in m."""
        return self.emittance_y * self.relativistic_beta * self.relativistic_gamma

    @property
    def beta_y(self) -> torch.Tensor:
        """Beta function in y direction in meters."""
        return self.sigma_y**2 / self.emittance_y

    @property
    def alpha_y(self) -> torch.Tensor:
        """Alpha function in y direction, dimensionless."""
        return -self.sigma_ypy / self.emittance_y

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mu_x={self.mu_x}, mu_px={self.mu_px},"
            f" mu_y={self.mu_y}, mu_py={self.mu_py}, sigma_x={self.sigma_x},"
            f" sigma_px={self.sigma_px}, sigma_y={self.sigma_y},"
            f" sigma_py={self.sigma_py}, sigma_tau={self.sigma_tau},"
            f" sigma_p={self.sigma_p}, energy={self.energy}),"
            f" total_charge={self.total_charge})"
        )
