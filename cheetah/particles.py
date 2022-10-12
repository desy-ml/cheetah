import numpy as np
import torch
from torch.distributions import MultivariateNormal

from cheetah.utils import from_astrabeam


class Beam:

    empty = "I'm an empty beam!"

    @classmethod
    def from_parameters(
        cls,
        mu_x=0,
        mu_xp=0,
        mu_y=0,
        mu_yp=0,
        sigma_x=175e-9,
        sigma_xp=2e-7,
        sigma_y=175e-9,
        sigma_yp=2e-7,
        sigma_s=1e-6,
        sigma_p=1e-6,
        cor_x=0,
        cor_y=0,
        cor_s=0,
        energy=1e8,
    ):
        """
        Create beam that with given beam parameters.

        Parameters
        ----------
        n : int, optional
            Number of particles to generate.
        mu_x : float, optional
            Center of the particle distribution on x in meters.
        mu_xp : float, optional
            Center of the particle distribution on px in meters.
        mu_y : float, optional
            Center of the particle distribution on y in meters.
        mu_yp : float, optional
            Center of the particle distribution on py in meters.
        sigma_x : float, optional
            Sgima of the particle distribution in x direction in meters.
        sigma_xp : float, optional
            Sgima of the particle distribution in px direction in meters.
        sigma_y : float, optional
            Sgima of the particle distribution in y direction in meters.
        sigma_yp : float, optional
            Sgima of the particle distribution in py direction in meters.
        sigma_s : float, optional
            Sgima of the particle distribution in s direction in meters.
        sigma_p : float, optional
            Sgima of the particle distribution in p direction in meters.
        energy : float, optional
            Energy of the beam in eV.
        """
        raise NotImplementedError

    @classmethod
    def from_ocelot(cls, parray):
        """
        Convert an Ocelot ParticleArray `parray` to a Cheetah Beam.
        """
        raise NotImplementedError

    @classmethod
    def from_astra(cls, path, **kwargs):
        """Load an Astra particle distribution as a Cheetah Beam."""
        raise NotImplementedError

    def transformed_to(
        self,
        mu_x=None,
        mu_xp=None,
        mu_y=None,
        mu_yp=None,
        sigma_x=None,
        sigma_xp=None,
        sigma_y=None,
        sigma_yp=None,
        sigma_s=None,
        sigma_p=None,
        energy=None,
    ):
        """
        Create version of this beam that is transformed to new beam parameters.

        Parameters
        ----------
        n : int, optional
            Number of particles to generate.
        mu_x : float, optional
            Center of the particle distribution on x in meters.
        mu_xp : float, optional
            Center of the particle distribution on px in meters.
        mu_y : float, optional
            Center of the particle distribution on y in meters.
        mu_yp : float, optional
            Center of the particle distribution on py in meters.
        sigma_x : float, optional
            Sgima of the particle distribution in x direction in meters.
        sigma_xp : float, optional
            Sgima of the particle distribution in px direction in meters.
        sigma_y : float, optional
            Sgima of the particle distribution in y direction in meters.
        sigma_yp : float, optional
            Sgima of the particle distribution in py direction in meters.
        sigma_s : float, optional
            Sgima of the particle distribution in s direction in meters.
        sigma_p : float, optional
            Sgima of the particle distribution in p direction in meters.
        energy : float, optional
            Energy of the beam in eV.
        """
        mu_x = mu_x if mu_x != None else self.mu_x
        mu_xp = mu_xp if mu_xp != None else self.mu_xp
        mu_y = mu_y if mu_y != None else self.mu_y
        mu_yp = mu_yp if mu_yp != None else self.mu_yp
        sigma_x = sigma_x if sigma_x != None else self.sigma_x
        sigma_xp = sigma_xp if sigma_xp != None else self.sigma_xp
        sigma_y = sigma_y if sigma_y != None else self.sigma_y
        sigma_yp = sigma_yp if sigma_yp != None else self.sigma_yp
        sigma_s = sigma_s if sigma_s != None else self.sigma_s
        sigma_p = sigma_p if sigma_p != None else self.sigma_p
        energy = energy if energy != None else self.energy

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
    def parameters(self):
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
    def mu_x(self):
        raise NotImplementedError

    @property
    def sigma_x(self):
        raise NotImplementedError

    @property
    def mu_xp(self):
        raise NotImplementedError

    @property
    def sigma_xp(self):
        raise NotImplementedError

    @property
    def mu_y(self):
        raise NotImplementedError

    @property
    def sigma_y(self):
        raise NotImplementedError

    @property
    def mu_yp(self):
        raise NotImplementedError

    @property
    def sigma_yp(self):
        raise NotImplementedError

    @property
    def mu_s(self):
        raise NotImplementedError

    @property
    def sigma_s(self):
        raise NotImplementedError

    @property
    def mu_p(self):
        raise NotImplementedError

    @property
    def sigma_p(self):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(mu_x={self.mu_x}, mu_xp={self.mu_xp}, mu_y={self.mu_y}, mu_yp={self.mu_yp}, sigma_x={self.sigma_x}, sigma_xp={self.sigma_xp}, sigma_y={self.sigma_y}, sigma_yp={self.sigma_yp}, sigma_s={self.sigma_s}, sigma_p={self.sigma_p}, energy={self.energy})"


class ParameterBeam(Beam):
    """
    Beam of charged particles, where each particle is simulated.

    Parameters
    ----------
    mu : torch.Tensor
        Mu vector of the beam.
    cov : torch.Tensor
        Covariance matrix of the beam.
    energy : float
        Energy of the beam in eV.
    """

    def __init__(self, mu, cov, energy):
        self._mu = mu
        self._cov = cov
        self.energy = energy

    @classmethod
    def from_parameters(
        cls,
        mu_x=0,
        mu_xp=0,
        mu_y=0,
        mu_yp=0,
        sigma_x=175e-9,
        sigma_xp=2e-7,
        sigma_y=175e-9,
        sigma_yp=2e-7,
        sigma_s=1e-6,
        sigma_p=1e-6,
        cor_x=0,
        cor_y=0,
        cor_s=0,
        energy=1e8,
    ):
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
    def from_ocelot(cls, parray):
        mu = torch.ones(7)
        mu[:6] = torch.tensor(parray.rparticles.mean(axis=1), dtype=torch.float32)

        cov = torch.zeros(7, 7)
        cov[:6, :6] = torch.tensor(np.cov(parray.rparticles), dtype=torch.float32)

        energy = 1e9 * parray.E

        return cls(mu=mu, cov=cov, energy=energy)

    @classmethod
    def from_astra(cls, path, **kwargs):
        """Load an Astra particle distribution as a Cheetah Beam."""
        particles, energy = from_astrabeam(path)
        mu = torch.ones(7)
        mu[:6] = torch.tensor(particles.mean(axis=0), dtype=torch.float32)

        cov = torch.zeros(7, 7)
        cov[:6, :6] = torch.tensor(np.cov(particles.transpose()), dtype=torch.float32)

        return cls(mu=mu, cov=cov, energy=energy)

    def transformed_to(
        self,
        mu_x=None,
        mu_xp=None,
        mu_y=None,
        mu_yp=None,
        sigma_x=None,
        sigma_xp=None,
        sigma_y=None,
        sigma_yp=None,
        sigma_s=None,
        sigma_p=None,
        energy=None,
    ):
        """
        Create version of this beam that is transformed to new beam parameters.

        Parameters
        ----------
        n : int, optional
            Number of particles to generate.
        mu_x : float, optional
            Center of the particle distribution on x in meters.
        mu_xp : float, optional
            Center of the particle distribution on px in meters.
        mu_y : float, optional
            Center of the particle distribution on y in meters.
        mu_yp : float, optional
            Center of the particle distribution on py in meters.
        sigma_x : float, optional
            Sgima of the particle distribution in x direction in meters.
        sigma_xp : float, optional
            Sgima of the particle distribution in px direction in meters.
        sigma_y : float, optional
            Sgima of the particle distribution in y direction in meters.
        sigma_yp : float, optional
            Sgima of the particle distribution in py direction in meters.
        sigma_s : float, optional
            Sgima of the particle distribution in s direction in meters.
        sigma_p : float, optional
            Sgima of the particle distribution in p direction in meters.
        energy : float, optional
            Energy of the beam in eV.
        """
        mu_x = mu_x if mu_x != None else self.mu_x
        mu_xp = mu_xp if mu_xp != None else self.mu_xp
        mu_y = mu_y if mu_y != None else self.mu_y
        mu_yp = mu_yp if mu_yp != None else self.mu_yp
        sigma_x = sigma_x if sigma_x != None else self.sigma_x
        sigma_xp = sigma_xp if sigma_xp != None else self.sigma_xp
        sigma_y = sigma_y if sigma_y != None else self.sigma_y
        sigma_yp = sigma_yp if sigma_yp != None else self.sigma_yp
        sigma_s = sigma_s if sigma_s != None else self.sigma_s
        sigma_p = sigma_p if sigma_p != None else self.sigma_p
        energy = energy if energy != None else self.energy

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
    def mu_x(self):
        return self._mu[0]

    @property
    def sigma_x(self):
        return torch.sqrt(self._cov[0, 0])

    @property
    def mu_xp(self):
        return self._mu[1]

    @property
    def sigma_xp(self):
        return torch.sqrt(self._cov[1, 1])

    @property
    def mu_y(self):
        return self._mu[2]

    @property
    def sigma_y(self):
        return torch.sqrt(self._cov[2, 2])

    @property
    def mu_yp(self):
        return self._mu[3]

    @property
    def sigma_yp(self):
        return torch.sqrt(self._cov[3, 3])

    @property
    def mu_s(self):
        return self._mu[4]

    @property
    def sigma_s(self):
        return torch.sqrt(self._cov[4, 4])

    @property
    def mu_p(self):
        return self._mu[5]

    @property
    def sigma_p(self):
        return torch.sqrt(self._cov[5, 5])

    def __repr__(self):
        return f"{self.__class__.__name__}(mu_x={self.mu_x:.6f}, mu_xp={self.mu_xp:.6f}, mu_y={self.mu_y:.6f}, mu_yp={self.mu_yp:.6f}, sigma_x={self.sigma_x:.6f}, sigma_xp={self.sigma_xp:.6f}, sigma_y={self.sigma_y:.6f}, sigma_yp={self.sigma_yp:.6f}, sigma_s={self.sigma_s:.6f}, sigma_p={self.sigma_p:.6f}, energy={self.energy:.3f})"


class ParticleBeam:
    """
    Beam of charged particles, where each particle is simulated.

    Parameters
    ----------
    particles : torch.Tensor
        List of 7-dimensional particle vectors.
    energy : float
        Energy of the beam in eV.
    device : string
        Device to move the beam's particle array to. If set to `"auto"` a CUDA GPU is
        selected if available. The CPU is used otherwise.
    """

    def __init__(self, particles, energy, device="auto"):
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
        n=100000,
        mu_x=0,
        mu_y=0,
        mu_xp=0,
        mu_yp=0,
        sigma_x=175e-9,
        sigma_y=175e-9,
        sigma_xp=2e-7,
        sigma_yp=2e-7,
        sigma_s=1e-6,
        sigma_p=1e-6,
        cor_x=0,
        cor_y=0,
        cor_s=0,
        energy=1e8,
        device="auto",
    ):
        """
        Generate Cheetah Beam of random particles.

        Parameters
        ----------
        n : int, optional
            Number of particles to generate.
        mu_x : float, optional
            Center of the particle distribution on x in meters.
        mu_y : float, optional
            Center of the particle distribution on y in meters.
        mu_xp : float, optional
            Center of the particle distribution on px in meters.
        mu_yp : float, optional
            Center of the particle distribution on py in meters.
        sigma_x : float, optional
            Sgima of the particle distribution in x direction in meters.
        sigma_y : float, optional
            Sgima of the particle distribution in y direction in meters.
        sigma_xp : float, optional
            Sgima of the particle distribution in px direction in meters.
        sigma_yp : float, optional
            Sgima of the particle distribution in py direction in meters.
        sigma_s : float, optional
            Sgima of the particle distribution in s direction in meters.
        sigma_p : float, optional
            Sgima of the particle distribution in p direction in meters.
        cor_x : float, optional
            Correlation between x and xp.
        cor_y : float, optional
            Correlation between y and yp.
        cor_s : float, optional
            Correlation between s and p.
        energy : float, optional
            Energy of the beam in eV.
        device : string
            Device to move the beam's particle array to. If set to `"auto"` a CUDA GPU
            is selected if available. The CPU is used otherwise.
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

        particles = torch.ones((n, 7), dtype=torch.float32)
        distribution = MultivariateNormal(mean, covariance_matrix=cov)
        particles[:, :6] = distribution.sample((n,))

        return cls(particles, energy, device=device)

    @classmethod
    def make_linspaced(
        cls,
        n=10,
        mu_x=0,
        mu_y=0,
        mu_xp=0,
        mu_yp=0,
        sigma_x=175e-9,
        sigma_y=175e-9,
        sigma_xp=2e-7,
        sigma_yp=2e-7,
        sigma_s=0,
        sigma_p=0,
        energy=1e8,
        device="auto",
    ):
        """
        Generate Cheetah Beam of *n* linspaced particles.

        Parameters
        ----------
        n : int, optional
            Number of particles to generate.
        mu_x : float, optional
            Center of the particle distribution on x in meters.
        mu_y : float, optional
            Center of the particle distribution on y in meters.
        mu_px : float, optional
            Center of the particle distribution on px in meters.
        mu_py : float, optional
            Center of the particle distribution on py in meters.
        sigma_x : float, optional
            Sgima of the particle distribution in x direction in meters.
        sigma_y : float, optional
            Sgima of the particle distribution in y direction in meters.
        sigma_xp : float, optional
            Sgima of the particle distribution in px direction in meters.
        sigma_yp : float, optional
            Sgima of the particle distribution in py direction in meters.
        sigma_s : float, optional
            Sgima of the particle distribution in s direction in meters.
        sigma_p : float, optional
            Sgima of the particle distribution in p direction in meters.
        energy : float, optional
            Energy of the beam in eV.
        device : string
            Device to move the beam's particle array to. If set to `"auto"` a CUDA GPU
            is selected if available. The CPU is used otherwise.
        """
        particles = torch.ones((n, 7), dtype=torch.float32)

        particles[:, 0] = torch.linspace(
            mu_x - sigma_x, mu_x + sigma_x, n, dtype=torch.float32
        )
        particles[:, 1] = torch.linspace(
            mu_xp - sigma_xp, mu_xp + sigma_xp, n, dtype=torch.float32
        )
        particles[:, 2] = torch.linspace(
            mu_y - sigma_y, mu_y + sigma_y, n, dtype=torch.float32
        )
        particles[:, 3] = torch.linspace(
            mu_yp - sigma_yp, mu_yp + sigma_yp, n, dtype=torch.float32
        )
        particles[:, 4] = torch.linspace(-sigma_s, sigma_s, n, dtype=torch.float32)
        particles[:, 5] = torch.linspace(-sigma_p, sigma_p, n, dtype=torch.float32)

        return cls(particles, energy, device=device)

    @classmethod
    def from_ocelot(cls, parray, device="auto"):
        """
        Convert an Ocelot ParticleArray `parray` to a Cheetah Beam.
        """
        n = parray.rparticles.shape[1]
        particles = torch.ones((n, 7))
        particles[:, :6] = torch.tensor(
            parray.rparticles.transpose(), dtype=torch.float32
        )

        return cls(particles, 1e9 * parray.E, device=device)

    @classmethod
    def from_astra(cls, path, **kwargs):
        """Load an Astra particle distribution as a Cheetah Beam."""
        particles, energy = from_astrabeam(path)
        particles_7d = torch.ones((particles.shape[0], 7))
        particles_7d[:, :6] = torch.from_numpy(particles)
        return cls(particles_7d, energy, **kwargs)

    def transformed_to(
        self,
        mu_x=None,
        mu_y=None,
        mu_xp=None,
        mu_yp=None,
        sigma_x=None,
        sigma_y=None,
        sigma_xp=None,
        sigma_yp=None,
        sigma_s=None,
        sigma_p=None,
        energy=None,
    ):
        """
        Create version of this beam that is transformed to new beam parameters.

        Parameters
        ----------
        n : int, optional
            Number of particles to generate.
        mu_x : float, optional
            Center of the particle distribution on x in meters.
        mu_y : float, optional
            Center of the particle distribution on y in meters.
        mu_xp : float, optional
            Center of the particle distribution on px in meters.
        mu_yp : float, optional
            Center of the particle distribution on py in meters.
        sigma_x : float, optional
            Sgima of the particle distribution in x direction in meters.
        sigma_y : float, optional
            Sgima of the particle distribution in y direction in meters.
        sigma_xp : float, optional
            Sgima of the particle distribution in px direction in meters.
        sigma_yp : float, optional
            Sgima of the particle distribution in py direction in meters.
        sigma_s : float, optional
            Sgima of the particle distribution in s direction in meters.
        sigma_p : float, optional
            Sgima of the particle distribution in p direction in meters.
        energy : float, optional
            Energy of the beam in eV.
        device : string
            Device to move the beam's particle array to. If set to `"auto"` a CUDA GPU
            is selected if available. The CPU is used otherwise.
        """
        mu_x = mu_x if mu_x != None else self.mu_x
        mu_y = mu_y if mu_y != None else self.mu_y
        mu_xp = mu_xp if mu_xp != None else self.mu_xp
        mu_yp = mu_yp if mu_yp != None else self.mu_yp
        sigma_x = sigma_x if sigma_x != None else self.sigma_x
        sigma_y = sigma_y if sigma_y != None else self.sigma_y
        sigma_xp = sigma_xp if sigma_xp != None else self.sigma_xp
        sigma_yp = sigma_yp if sigma_yp != None else self.sigma_yp
        sigma_s = sigma_s if sigma_s != None else self.sigma_s
        sigma_p = sigma_p if sigma_p != None else self.sigma_p
        energy = energy if energy != None else self.energy

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

    def __len__(self):
        return self.n

    @property
    def n(self):
        return len(self.particles)

    @property
    def xs(self):
        return self.particles[:, 0] if self is not Beam.empty else None

    @xs.setter
    def xs(self, value):
        self.particles[:, 0] = value

    @property
    def mu_x(self):
        return float(self.xs.mean()) if self is not Beam.empty else None

    @property
    def sigma_x(self):
        return float(self.xs.std()) if self is not Beam.empty else None

    @property
    def xps(self):
        return self.particles[:, 1] if self is not Beam.empty else None

    @xps.setter
    def xps(self, value):
        self.particles[:, 1] = value

    @property
    def mu_xp(self):
        return float(self.xps.mean()) if self is not Beam.empty else None

    @property
    def sigma_xp(self):
        return float(self.xps.std()) if self is not Beam.empty else None

    @property
    def ys(self):
        return self.particles[:, 2] if self is not Beam.empty else None

    @ys.setter
    def ys(self, value):
        self.particles[:, 2] = value

    @property
    def mu_y(self):
        return float(self.ys.mean()) if self is not Beam.empty else None

    @property
    def sigma_y(self):
        return float(self.ys.std()) if self is not Beam.empty else None

    @property
    def yps(self):
        return self.particles[:, 3] if self is not Beam.empty else None

    @yps.setter
    def yps(self, value):
        self.particles[:, 3] = value

    @property
    def mu_yp(self):
        return float(self.yps.mean()) if self is not Beam.empty else None

    @property
    def sigma_yp(self):
        return float(self.yps.std()) if self is not Beam.empty else None

    @property
    def ss(self):
        return self.particles[:, 4] if self is not Beam.empty else None

    @ss.setter
    def ss(self, value):
        self.particles[:, 4] = value

    @property
    def mu_s(self):
        return float(self.ss.mean()) if self is not Beam.empty else None

    @property
    def sigma_s(self):
        return float(self.ss.std()) if self is not Beam.empty else None

    @property
    def ps(self):
        return self.particles[:, 5] if self is not Beam.empty else None

    @ps.setter
    def ps(self, value):
        self.particles[:, 5] = value

    @property
    def mu_p(self):
        return float(self.ps.mean()) if self is not Beam.empty else None

    @property
    def sigma_p(self):
        return float(self.ps.std()) if self is not Beam.empty else None

    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.n}, mu_x={self.mu_x:.6f}, mu_xp={self.mu_xp:.6f}, mu_y={self.mu_y:.6f}, mu_yp={self.mu_yp:.6f}, sigma_x={self.sigma_x:.6f}, sigma_xp={self.sigma_xp:.6f}, sigma_y={self.sigma_y:.6f}, sigma_yp={self.sigma_yp:.6f}, sigma_s={self.sigma_s:.6f}, sigma_p={self.sigma_p:.6f}, energy={self.energy:.3f})"
