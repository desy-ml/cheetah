from abc import ABC, abstractmethod

import torch
from torch import nn

from cheetah.particles.species import Species


class Beam(ABC, nn.Module):
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

    UNVECTORIZED_NUM_ATTR_DIMS = {
        "energy": 0,
        "total_charge": 0,
        "s": 0,
        "mu_x": 0,
        "sigma_x": 0,
        "mu_px": 0,
        "sigma_px": 0,
        "mu_y": 0,
        "sigma_y": 0,
        "mu_py": 0,
        "sigma_py": 0,
        "mu_tau": 0,
        "sigma_tau": 0,
        "mu_p": 0,
        "sigma_p": 0,
        "relativistic_gamma": 0,
        "relativistic_beta": 0,
        "p0c": 0,
        "cov_xpx": 0,
        "cov_ypy": 0,
        "cov_taup": 0,
        "emittance_x": 0,
        "normalized_emittance_x": 0,
        "beta_x": 0,
        "alpha_x": 0,
        "emittance_y": 0,
        "normalized_emittance_y": 0,
        "beta_y": 0,
        "alpha_y": 0,
    }

    @classmethod
    @abstractmethod
    def from_parameters(
        cls,
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
    ) -> "Beam":
        """
        Create beam that with given beam parameters.

        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_px: Center of the particle distribution on px, dimensionless.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_py: Center of the particle distribution on yp, dimensionless.
        :param mu_tau: Center of the particle distribution on tau in meters.
        :param mu_p: Center of the particle distribution on p, dimensionless.
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
        :param cov_xpx: Covariance between x and px.
        :param cov_ypy: Covariance between y and yp.
        :param cov_taup: Covariance between tau and p.
        :param energy: Reference energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        :param s: Position along the beamline of the reference particle in meters.
        :param species: Particle species of the beam. Defaults to electron.
        :param device: Device to create the beam on. If set to `"auto"` a CUDA GPU is
            selected if available. The CPU is used otherwise.
        :param dtype: Data type of the beam.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_twiss(
        cls,
        beta_x: torch.Tensor | None = None,
        alpha_x: torch.Tensor | None = None,
        emittance_x: torch.Tensor | None = None,
        beta_y: torch.Tensor | None = None,
        alpha_y: torch.Tensor | None = None,
        emittance_y: torch.Tensor | None = None,
        sigma_tau: torch.Tensor | None = None,
        sigma_p: torch.Tensor | None = None,
        cov_taup: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
        total_charge: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        species: Species | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Beam":
        """
        Create a beam from twiss parameters.

        :param beta_x: Beta function in x direction in meters.
        :param alpha_x: Alpha function in x direction in rad.
        :param emittance_x: Emittance in x direction in m*rad.
        :param beta_y: Beta function in y direction in meters.
        :param alpha_y: Alpha function in y direction in rad.
        :param emittance_y: Emittance in y direction in m*rad.
        :param sigma_tau: Sigma of the particle distribution in longitudinal direction,
            in meters.
        :param sigma_p: Sigma of the particle distribution in p direction,
            dimensionless.
        :param cov_taup: Covariance between tau and p.
        :param energy: Energy of the beam in eV.
        :param total_charge: Total charge of the beam in C.
        :param species: Particle species of the beam. Defaults to electron.
        :param s: Position along the beamline of the reference particle in meters.
        :param device: Device to create the beam on. If set to `"auto"` a CUDA GPU is
            selected if available. The CPU is used otherwise.
        :param dtype: Data type of the beam.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_ocelot(
        cls, parray, device: torch.device = None, dtype: torch.dtype = None
    ) -> "Beam":
        """Convert an Ocelot ParticleArray `parray` to a Cheetah Beam."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_astra(
        cls, path: str, device: torch.device = None, dtype: torch.dtype = None
    ) -> "Beam":
        """Load an Astra particle distribution as a Cheetah Beam."""
        raise NotImplementedError

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
    ) -> "Beam":
        """
        Create version of this beam that is transformed to new beam parameters.

        :param mu_x: Center of the particle distribution on x in meters.
        :param mu_px: Center of the particle distribution on px, dimensionless.
        :param mu_y: Center of the particle distribution on y in meters.
        :param mu_py: Center of the particle distribution on yp, dimensionless.
        :param mu_tau: Center of the particle distribution on tau in meters.
        :param mu_p: Center of the particle distribution on p, dimensionless.
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
        :param species: Particle species of the beam.
        :param device: Device to create the transformed beam on. If set to `"auto"` a
            CUDA GPU is selected if available. The CPU is used otherwise.
        :param dtype: Data type of the transformed beam.
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
                mu_tau,
                mu_p,
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
        mu_tau = mu_tau if mu_tau is not None else self.mu_tau
        mu_p = mu_p if mu_p is not None else self.mu_p
        sigma_x = sigma_x if sigma_x is not None else self.sigma_x
        sigma_px = sigma_px if sigma_px is not None else self.sigma_px
        sigma_y = sigma_y if sigma_y is not None else self.sigma_y
        sigma_py = sigma_py if sigma_py is not None else self.sigma_py
        sigma_tau = sigma_tau if sigma_tau is not None else self.sigma_tau
        sigma_p = sigma_p if sigma_p is not None else self.sigma_p
        energy = energy if energy is not None else self.energy
        total_charge = total_charge if total_charge is not None else self.total_charge
        species = species if species is not None else self.species

        return self.__class__.from_parameters(
            mu_x=mu_x,
            mu_px=mu_px,
            mu_y=mu_y,
            mu_py=mu_py,
            mu_tau=mu_tau,
            mu_p=mu_p,
            sigma_x=sigma_x,
            sigma_px=sigma_px,
            sigma_y=sigma_y,
            sigma_py=sigma_py,
            sigma_tau=sigma_tau,
            sigma_p=sigma_p,
            energy=energy,
            total_charge=total_charge,
            species=species,
            device=device,
            dtype=dtype,
        )

    @property
    @abstractmethod
    def mu_x(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def sigma_x(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def mu_px(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def sigma_px(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def mu_y(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def sigma_y(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def mu_py(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def sigma_py(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def mu_tau(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def sigma_tau(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def mu_p(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def sigma_p(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def relativistic_gamma(self) -> torch.Tensor:
        """Reference relativistic gamma of the beam."""
        return self.energy / self.species.mass_eV

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
        """Reference momentum * speed of light in eV."""
        return self.relativistic_beta * self.relativistic_gamma * self.species.mass_eV

    @property
    @abstractmethod
    def cov_xpx(self) -> torch.Tensor:
        # The covariance of (x,px) ~ $\sigma_{xpx}$
        raise NotImplementedError

    @property
    @abstractmethod
    def cov_ypy(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def cov_taup(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def emittance_x(self) -> torch.Tensor:
        """Emittance of the beam in x direction in m."""
        return torch.sqrt(
            torch.clamp_min(
                self.sigma_x**2 * self.sigma_px**2 - self.cov_xpx**2,
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
        return -self.cov_xpx / self.emittance_x

    @property
    def emittance_y(self) -> torch.Tensor:
        """Emittance of the beam in y direction in m."""
        return torch.sqrt(
            torch.clamp_min(
                self.sigma_y**2 * self.sigma_py**2 - self.cov_ypy**2,
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
        return -self.cov_ypy / self.emittance_y

    @abstractmethod
    def clone(self) -> "Beam":
        """Return a cloned beam that does not share the underlying memory."""
        raise NotImplementedError

    def register_buffer_or_parameter(
        self, name: str, value: torch.Tensor | nn.Parameter
    ) -> None:
        """
        Register a buffer or parameter with the given name and value. Automatically
        selects the correct method from `register_buffer` or `register_parameter` based
        on the type of `value`.

        :param name: Name of the buffer or parameter.
        :param value: Value of the buffer or parameter.
        :param default: Default value of the buffer.
        """
        if isinstance(value, nn.Parameter):
            self.register_parameter(name, value)
        else:
            self.register_buffer(name, value)

    def __repr__(self) -> str:
        raise NotImplementedError
