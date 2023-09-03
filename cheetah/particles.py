from typing import Optional

import numpy as np
import torch
from scipy.constants import physical_constants
from torch.distributions import MultivariateNormal

electron_mass_eV = torch.tensor(
    physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
)


class Beam:
    empty = "I'm an empty beam!"

    @classmethod
    def from_parameters(
        cls,
        mu_x: float = 0,
        mu_xp: float = 0,
        mu_y: float = 0,
        mu_yp: float = 0,
        sigma_x: float = 1,
        sigma_xp: float = 0,
        sigma_y: float = 1,
        sigma_yp: float = 0,
        sigma_s: float = 0,
        sigma_p: float = 0,
        cor_x: float = 0,
        cor_y: float = 0,
        cor_s: float = 0,
        energy: float = 1e8,
    ) -> "Beam":
        """
        Create beam that with given beam parameters.

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
        mu_x: Optional[float] = None,
        mu_xp: Optional[float] = None,
        mu_y: Optional[float] = None,
        mu_yp: Optional[float] = None,
        sigma_x: Optional[float] = None,
        sigma_xp: Optional[float] = None,
        sigma_y: Optional[float] = None,
        sigma_yp: Optional[float] = None,
        sigma_s: Optional[float] = None,
        sigma_p: Optional[float] = None,
        energy: Optional[float] = None,
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
        """
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
    def emittance_x(self) -> torch.Tensor:
        return torch.sqrt(
            self.sigma_x**2 * self.sigma_xp**2
            - torch.mean((self.xs - self.mu_x) * (self.xps - self.mu_xp)) ** 2
        )

    @property
    def normalized_emittance_x(self) -> torch.Tensor:
        relativistic_gamma = self.energy / electron_mass_eV
        relativistic_beta = (
            torch.sqrt(1 - 1 / (relativistic_gamma**2))
            if torch.abs(relativistic_gamma) > 0
            else 1.0
        )
        return self.emittance_x * relativistic_beta * relativistic_gamma

    @property
    def beta_x(self) -> torch.Tensor:
        return self.sigma_x**2 / self.emittance_x

    @property
    def alpha_x(self) -> torch.Tensor:
        return self.sigma_xp**2 / self.emittance_x  # TODO: Does this make sense?

    @property
    def emittance_y(self) -> torch.Tensor:
        return torch.sqrt(
            self.sigma_y**2 * self.sigma_yp**2
            - torch.mean((self.ys - self.mu_y) * (self.yps - self.mu_yp)) ** 2
        )

    @property
    def normalized_emittance_y(self) -> torch.Tensor:
        relativistic_gamma = self.energy / electron_mass_eV
        relativistic_beta = (
            torch.sqrt(1 - 1 / (relativistic_gamma**2))
            if torch.abs(relativistic_gamma) > 0
            else 1.0
        )
        return self.emittance_y * relativistic_beta * relativistic_gamma

    @property
    def beta_y(self) -> torch.Tensor:
        return self.sigma_y**2 / self.emittance_y

    @property
    def alpha_y(self) -> torch.Tensor:
        return self.sigma_yp**2 / self.emittance_y  # TODO: Does this make sense?

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mu_x={self.mu_x}, mu_xp={self.mu_xp},"
            f" mu_y={self.mu_y}, mu_yp={self.mu_yp}, sigma_x={self.sigma_x},"
            f" sigma_xp={self.sigma_xp}, sigma_y={self.sigma_y},"
            f" sigma_yp={self.sigma_yp}, sigma_s={self.sigma_s},"
            f" sigma_p={self.sigma_p}, energy={self.energy})"
        )


class ParameterBeam(Beam):
    """
    Beam of charged particles, where each particle is simulated.

    :param mu: Mu vector of the beam.
    :param cov: Covariance matrix of the beam.
    :param energy: Energy of the beam in eV.
    """

    def __init__(self, mu: torch.Tensor, cov: torch.Tensor, energy: float) -> None:
        self._mu = mu
        self._cov = cov
        self.energy = energy

    @classmethod
    def from_parameters(
        cls,
        mu_x: float = 0,
        mu_xp: float = 0,
        mu_y: float = 0,
        mu_yp: float = 0,
        sigma_x: float = 175e-9,
        sigma_xp: float = 2e-7,
        sigma_y: float = 175e-9,
        sigma_yp: float = 2e-7,
        sigma_s: float = 1e-6,
        sigma_p: float = 1e-6,
        cor_x: float = 0,
        cor_y: float = 0,
        cor_s: float = 0,
        energy: float = 1e8,
    ) -> "ParameterBeam":
        return cls(
            mu=torch.tensor([mu_x, mu_xp, mu_y, mu_yp, 0, 0, 1], dtype=torch.float32),
            cov=torch.tensor(
                [
                    [sigma_x**2, cor_x, 0, 0, 0, 0, 0],
                    [cor_x, sigma_xp**2, 0, 0, 0, 0, 0],
                    [0, 0, sigma_y**2, cor_y, 0, 0, 0],
                    [0, 0, cor_y, sigma_yp**2, 0, 0, 0],
                    [0, 0, 0, 0, sigma_s**2, cor_s, 0],
                    [0, 0, 0, 0, cor_s, sigma_p**2, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=torch.float32,
            ),
            energy=energy,
        )

    @classmethod
    def from_ocelot(cls, parray) -> "ParameterBeam":
        mu = torch.ones(7)
        mu[:6] = torch.tensor(parray.rparticles.mean(axis=1), dtype=torch.float32)

        cov = torch.zeros(7, 7)
        cov[:6, :6] = torch.tensor(np.cov(parray.rparticles), dtype=torch.float32)

        energy = 1e9 * parray.E

        return cls(mu=mu, cov=cov, energy=energy)

    @classmethod
    def from_astra(cls, path: str, **kwargs) -> "ParameterBeam":
        """Load an Astra particle distribution as a Cheetah Beam."""
        from cheetah.utils import from_astrabeam

        particles, energy = from_astrabeam(path)
        mu = torch.ones(7)
        mu[:6] = torch.tensor(particles.mean(axis=0), dtype=torch.float32)

        cov = torch.zeros(7, 7)
        cov[:6, :6] = torch.tensor(np.cov(particles.transpose()), dtype=torch.float32)

        return cls(mu=mu, cov=cov, energy=energy)

    def transformed_to(
        self,
        mu_x: Optional[float] = None,
        mu_xp: Optional[float] = None,
        mu_y: Optional[float] = None,
        mu_yp: Optional[float] = None,
        sigma_x: Optional[float] = None,
        sigma_xp: Optional[float] = None,
        sigma_y: Optional[float] = None,
        sigma_yp: Optional[float] = None,
        sigma_s: Optional[float] = None,
        sigma_p: Optional[float] = None,
        energy: Optional[float] = None,
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
        """
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
        )

    @property
    def mu_x(self) -> torch.Tensor:
        return self._mu[0]

    @property
    def sigma_x(self) -> torch.Tensor:
        return torch.sqrt(self._cov[0, 0])

    @property
    def mu_xp(self) -> torch.Tensor:
        return self._mu[1]

    @property
    def sigma_xp(self) -> torch.Tensor:
        return torch.sqrt(self._cov[1, 1])

    @property
    def mu_y(self) -> torch.Tensor:
        return self._mu[2]

    @property
    def sigma_y(self) -> torch.Tensor:
        return torch.sqrt(self._cov[2, 2])

    @property
    def mu_yp(self) -> torch.Tensor:
        return self._mu[3]

    @property
    def sigma_yp(self) -> torch.Tensor:
        return torch.sqrt(self._cov[3, 3])

    @property
    def mu_s(self) -> torch.Tensor:
        return self._mu[4]

    @property
    def sigma_s(self) -> torch.Tensor:
        return torch.sqrt(self._cov[4, 4])

    @property
    def mu_p(self) -> torch.Tensor:
        return self._mu[5]

    @property
    def sigma_p(self) -> torch.Tensor:
        return torch.sqrt(self._cov[5, 5])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mu_x={self.mu_x:.6f}, mu_xp={self.mu_xp:.6f},"
            f" mu_y={self.mu_y:.6f}, mu_yp={self.mu_yp:.6f},"
            f" sigma_x={self.sigma_x:.6f}, sigma_xp={self.sigma_xp:.6f},"
            f" sigma_y={self.sigma_y:.6f}, sigma_yp={self.sigma_yp:.6f},"
            f" sigma_s={self.sigma_s:.6f}, sigma_p={self.sigma_p:.6f},"
            f" energy={self.energy:.3f})"
        )


class ParticleBeam(Beam):
    """
    Beam of charged particles, where each particle is simulated.

    :param particles: List of 7-dimensional particle vectors.
    :param energy: Energy of the beam in eV.
    :param device: Device to move the beam's particle array to. If set to `"auto"` a
        CUDA GPU is selected if available. The CPU is used otherwise.
    """

    def __init__(
        self, particles: torch.Tensor, energy: float, device: str = "auto"
    ) -> None:
        assert (
            len(particles) > 0 and particles.shape[1] == 7
        ), "Particle vectors must be 7-dimensional."

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.particles = (
            particles.to(device) if isinstance(particles, torch.Tensor) else particles
        )

        self.energy = energy

    @classmethod
    def from_parameters(
        cls,
        num_particles: int = 100000,
        mu_x: float = 0,
        mu_y: float = 0,
        mu_xp: float = 0,
        mu_yp: float = 0,
        sigma_x: float = 175e-9,
        sigma_y: float = 175e-9,
        sigma_xp: float = 2e-7,
        sigma_yp: float = 2e-7,
        sigma_s: float = 1e-6,
        sigma_p: float = 1e-6,
        cor_x: float = 0,
        cor_y: float = 0,
        cor_s: float = 0,
        energy: float = 1e8,
        device: str = "auto",
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
        :param device: Device to move the beam's particle array to. If set to `"auto"` a
            CUDA GPU is selected if available. The CPU is used otherwise.
        """
        mean = torch.tensor([mu_x, mu_xp, mu_y, mu_yp, 0, 0], dtype=torch.float32)
        cov = torch.tensor(
            [
                [sigma_x**2, cor_x, 0, 0, 0, 0],
                [cor_x, sigma_xp**2, 0, 0, 0, 0],
                [0, 0, sigma_y**2, cor_y, 0, 0],
                [0, 0, cor_y, sigma_yp**2, 0, 0],
                [0, 0, 0, 0, sigma_s**2, cor_s],
                [0, 0, 0, 0, cor_s, sigma_p**2],
            ],
            dtype=torch.float32,
        )

        particles = torch.ones((num_particles, 7), dtype=torch.float32)
        distribution = MultivariateNormal(mean, covariance_matrix=cov)
        particles[:, :6] = distribution.sample((num_particles,))

        return cls(particles, energy, device=device)

    @classmethod
    def make_linspaced(
        cls,
        num_particles: int = 10,
        mu_x: float = 0,
        mu_y: float = 0,
        mu_xp: float = 0,
        mu_yp: float = 0,
        sigma_x: float = 175e-9,
        sigma_y: float = 175e-9,
        sigma_xp: float = 2e-7,
        sigma_yp: float = 2e-7,
        sigma_s: float = 0,
        sigma_p: float = 0,
        energy: float = 1e8,
        device: str = "auto",
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
        particles = torch.ones((num_particles, 7), dtype=torch.float32)

        particles[:, 0] = torch.linspace(
            mu_x - sigma_x, mu_x + sigma_x, num_particles, dtype=torch.float32
        )
        particles[:, 1] = torch.linspace(
            mu_xp - sigma_xp, mu_xp + sigma_xp, num_particles, dtype=torch.float32
        )
        particles[:, 2] = torch.linspace(
            mu_y - sigma_y, mu_y + sigma_y, num_particles, dtype=torch.float32
        )
        particles[:, 3] = torch.linspace(
            mu_yp - sigma_yp, mu_yp + sigma_yp, num_particles, dtype=torch.float32
        )
        particles[:, 4] = torch.linspace(
            -sigma_s, sigma_s, num_particles, dtype=torch.float32
        )
        particles[:, 5] = torch.linspace(
            -sigma_p, sigma_p, num_particles, dtype=torch.float32
        )

        return cls(particles, energy, device=device)

    @classmethod
    def from_ocelot(cls, parray, device: str = "auto") -> "ParticleBeam":
        """
        Convert an Ocelot ParticleArray `parray` to a Cheetah Beam.
        """
        num_particles = parray.rparticles.shape[1]
        particles = torch.ones((num_particles, 7))
        particles[:, :6] = torch.tensor(
            parray.rparticles.transpose(), dtype=torch.float32
        )

        return cls(particles, 1e9 * parray.E, device=device)

    @classmethod
    def from_astra(cls, path: str, **kwargs) -> "ParticleBeam":
        """Load an Astra particle distribution as a Cheetah Beam."""
        from cheetah.utils import from_astrabeam

        particles, energy = from_astrabeam(path)
        particles_7d = torch.ones((particles.shape[0], 7))
        particles_7d[:, :6] = torch.from_numpy(particles)
        return cls(particles_7d, energy, **kwargs)

    def transformed_to(
        self,
        mu_x: Optional[float] = None,
        mu_y: Optional[float] = None,
        mu_xp: Optional[float] = None,
        mu_yp: Optional[float] = None,
        sigma_x: Optional[float] = None,
        sigma_y: Optional[float] = None,
        sigma_xp: Optional[float] = None,
        sigma_yp: Optional[float] = None,
        sigma_s: Optional[float] = None,
        sigma_p: Optional[float] = None,
        energy: Optional[float] = None,
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
        :param device: Device to move the beam's particle array to. If set to `"auto"` a
            CUDA GPU is selected if available. The CPU is used otherwise.
        """
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

        new_mu = torch.tensor(
            [mu_x, mu_xp, mu_y, mu_yp, 0, 0], dtype=torch.float32, device=self.device
        )
        new_sigma = torch.tensor(
            [sigma_x, sigma_xp, sigma_y, sigma_yp, sigma_s, sigma_p],
            dtype=torch.float32,
            device=self.device,
        )

        old_mu = torch.tensor(
            [self.mu_x, self.mu_xp, self.mu_y, self.mu_yp, 0, 0],
            dtype=torch.float32,
            device=self.device,
        )
        old_sigma = torch.tensor(
            [
                self.sigma_x,
                self.sigma_xp,
                self.sigma_y,
                self.sigma_yp,
                self.sigma_s,
                self.sigma_p,
            ],
            dtype=torch.float32,
            device=self.device,
        )

        phase_space = self.particles[:, :6]
        phase_space = (phase_space - old_mu) / old_sigma * new_sigma + new_mu

        particles = torch.ones_like(
            self.particles, dtype=torch.float32, device=self.device
        )
        particles[:, :6] = phase_space

        return self.__class__(particles=particles, energy=energy)

    def __len__(self) -> int:
        return self.num_particles

    @property
    def num_particles(self) -> int:
        return len(self.particles)

    @property
    def xs(self) -> Optional[torch.Tensor]:
        return self.particles[:, 0] if self is not Beam.empty else None

    @xs.setter
    def xs(self, value: torch.Tensor) -> None:
        self.particles[:, 0] = value

    @property
    def mu_x(self) -> Optional[torch.Tensor]:
        return self.xs.mean() if self is not Beam.empty else None

    @property
    def sigma_x(self) -> Optional[torch.Tensor]:
        return self.xs.std() if self is not Beam.empty else None

    @property
    def xps(self) -> Optional[torch.Tensor]:
        return self.particles[:, 1] if self is not Beam.empty else None

    @xps.setter
    def xps(self, value: torch.Tensor) -> None:
        self.particles[:, 1] = value

    @property
    def mu_xp(self) -> Optional[torch.Tensor]:
        return self.xps.mean() if self is not Beam.empty else None

    @property
    def sigma_xp(self) -> Optional[torch.Tensor]:
        return self.xps.std() if self is not Beam.empty else None

    @property
    def ys(self) -> Optional[torch.Tensor]:
        return self.particles[:, 2] if self is not Beam.empty else None

    @ys.setter
    def ys(self, value: torch.Tensor) -> None:
        self.particles[:, 2] = value

    @property
    def mu_y(self) -> Optional[float]:
        return self.ys.mean() if self is not Beam.empty else None

    @property
    def sigma_y(self) -> Optional[torch.Tensor]:
        return self.ys.std() if self is not Beam.empty else None

    @property
    def yps(self) -> Optional[torch.Tensor]:
        return self.particles[:, 3] if self is not Beam.empty else None

    @yps.setter
    def yps(self, value: torch.Tensor) -> None:
        self.particles[:, 3] = value

    @property
    def mu_yp(self) -> Optional[torch.Tensor]:
        return self.yps.mean() if self is not Beam.empty else None

    @property
    def sigma_yp(self) -> Optional[torch.Tensor]:
        return self.yps.std() if self is not Beam.empty else None

    @property
    def ss(self) -> Optional[torch.Tensor]:
        return self.particles[:, 4] if self is not Beam.empty else None

    @ss.setter
    def ss(self, value: torch.Tensor) -> None:
        self.particles[:, 4] = value

    @property
    def mu_s(self) -> Optional[torch.Tensor]:
        return self.ss.mean() if self is not Beam.empty else None

    @property
    def sigma_s(self) -> Optional[torch.Tensor]:
        return self.ss.std() if self is not Beam.empty else None

    @property
    def ps(self) -> Optional[torch.Tensor]:
        return self.particles[:, 5] if self is not Beam.empty else None

    @ps.setter
    def ps(self, value: torch.Tensor) -> None:
        self.particles[:, 5] = value

    @property
    def mu_p(self) -> Optional[torch.Tensor]:
        return self.ps.mean() if self is not Beam.empty else None

    @property
    def sigma_p(self) -> Optional[torch.Tensor]:
        return self.ps.std() if self is not Beam.empty else None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n={self.num_particles}, mu_x={self.mu_x:.6f},"
            f" mu_xp={self.mu_xp:.6f}, mu_y={self.mu_y:.6f}, mu_yp={self.mu_yp:.6f},"
            f" sigma_x={self.sigma_x:.6f}, sigma_xp={self.sigma_xp:.6f},"
            f" sigma_y={self.sigma_y:.6f}, sigma_yp={self.sigma_yp:.6f},"
            f" sigma_s={self.sigma_s:.6f}, sigma_p={self.sigma_p:.6f},"
            f" energy={self.energy:.3f})"
        )
