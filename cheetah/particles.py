import torch
from torch.distributions import MultivariateNormal


class Beam:
    """
    Beam of charged particles.

    Parameters
    ----------
    particles : torch.Tensor
        List of 7-dimensional particle vectors.
    energy : float
        Energy of the beam in eV.
    device : string
        Device to move the beam's particle array to. If set to `"auto"` a CUDA GPU is selected if
        available. The CPU is used otherwise.
    """

    def __init__(self, particles, energy, device="auto"):
        assert len(particles) == 0 or particles.shape[1] == 7, "Particle vectors must be 7-dimensional."

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.particles = particles.to(device) if isinstance(particles, torch.Tensor) else particles
        
        self.energy = energy
    
    @classmethod
    def make_random(cls, n=100000, mu_x=0, mu_y=0, mu_xp=0, mu_yp=0, sigma_x=175e-9, sigma_y=175e-9,
                    sigma_xp=2e-7, sigma_yp=2e-7, sigma_s=1e-6, sigma_p=1e-6, cor_x=0, cor_y=0,
                    cor_s=0, energy=1e8, device="auto"):
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
        device : string
            Device to move the beam's particle array to. If set to `"auto"` a CUDA GPU is selected if
            available. The CPU is used otherwise.
        """
        mean = torch.tensor([mu_x, mu_xp, mu_y, mu_yp, 0, 0], dtype=torch.float32)
        cov = torch.tensor([[sigma_x**2,       cor_x,          0,           0,          0,          0],
                            [     cor_x, sigma_xp**2,          0,           0,          0,          0],
                            [         0,           0, sigma_y**2,       cor_y,          0,          0],
                            [         0,           0,      cor_y, sigma_yp**2,          0,          0],
                            [         0,           0,          0,           0, sigma_s**2,      cor_s],
                            [         0,           0,          0,           0,      cor_s, sigma_p**2]], dtype=torch.float32)

        particles = torch.ones((n, 7), dtype=torch.float32)
        distribution = MultivariateNormal(mean, covariance_matrix=cov)
        particles[:,:6] = distribution.sample((n,))

        return cls(particles, energy, device=device)
    
    
    @classmethod
    def make_linspaced(cls, n=10, mu_x=0, mu_y=0, mu_xp=0, mu_yp=0, sigma_x=175e-9, sigma_y=175e-9,
                       sigma_xp=2e-7, sigma_yp=2e-7, sigma_s=0, sigma_p=0, energy=1e8, device="auto"):
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
        device : string
            Device to move the beam's particle array to. If set to `"auto"` a CUDA GPU is selected if
            available. The CPU is used otherwise.
        """
        particles = torch.ones((n, 7), dtype=torch.float32)
        
        particles[:,0] = torch.linspace(mu_x-sigma_x, mu_x+sigma_x, n, dtype=torch.float32)
        particles[:,1] = torch.linspace(mu_xp-sigma_xp, mu_xp+sigma_xp, n, dtype=torch.float32)
        particles[:,2] = torch.linspace(mu_y-sigma_y, mu_y+sigma_y, n, dtype=torch.float32)
        particles[:,3] = torch.linspace(mu_yp-sigma_yp, mu_yp+sigma_yp, n, dtype=torch.float32)
        particles[:,4] = torch.linspace(-sigma_s, sigma_s, n, dtype=torch.float32)
        particles[:,5] = torch.linspace(-sigma_p, sigma_p, n, dtype=torch.float32)

        return cls(particles, energy, device=device)

    @classmethod
    def from_ocelot(cls, parray, device="auto"):
        """
        Convert an Ocelot ParticleArray `parray` to a Cheetah Beam.
        """
        n = parray.rparticles.shape[1]
        particles = torch.ones((n, 7))
        particles[:,:6] = torch.tensor(parray.rparticles.transpose(), dtype=torch.float32)

        return cls(particles, 1e9*parray.E, device=device)
    
    def __len__(self):
        return self.n
    
    @property
    def n(self):
        return len(self.particles)
    
    @property
    def is_empty(self):
        return self.n == 0
    
    @property
    def xs(self):
        return self.particles[:,0] if not self.is_empty else None
    
    @xs.setter
    def xs(self, value):
        self.particles[:,0] = value
    
    @property
    def mu_x(self):
        return self.xs.mean() if not self.is_empty else None
    
    @property
    def sigma_x(self):
        return self.xs.std() if not self.is_empty else None
    
    @property
    def xps(self):
        return self.particles[:,1] if not self.is_empty else None
    
    @xps.setter
    def xps(self, value):
        self.particles[:,1] = value
    
    @property
    def mu_xp(self):
        return self.xps.mean() if not self.is_empty else None
    
    @property
    def sigma_xp(self):
        return self.xps.std() if not self.is_empty else None
    
    @property
    def ys(self):
        return self.particles[:,2] if not self.is_empty else None
    
    @ys.setter
    def ys(self, value):
        self.particles[:,2] = value
    
    @property
    def mu_y(self):
        return self.ys.mean() if not self.is_empty else None
    
    @property
    def sigma_y(self):
        return self.ys.std() if not self.is_empty else None
    
    @property
    def yps(self):
        return self.particles[:,3] if not self.is_empty else None
    
    @yps.setter
    def yps(self, value):
        self.particles[:,3] = value
    
    @property
    def mu_yp(self):
        return self.yps.mean() if not self.is_empty else None
    
    @property
    def sigma_yp(self):
        return self.yps.std() if not self.is_empty else None
    
    @property
    def ss(self):
        return self.particles[:,4] if not self.is_empty else None
    
    @ss.setter
    def ss(self, value):
        self.particles[:,4] = value
    
    @property
    def mu_s(self):
        return self.ss.mean() if not self.is_empty else None
    
    @property
    def sigma_s(self):
        return self.ss.std() if not self.is_empty else None
    
    @property
    def ps(self):
        return self.particles[:,5] if not self.is_empty else None
    
    @ps.setter
    def ps(self, value):
        self.particles[:,5] = value
    
    @property
    def mu_p(self):
        return self.ps.mean() if not self.is_empty else None
    
    @property
    def sigma_p(self):
        return self.ps.std() if not self.is_empty else None
    
    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.n}, mu_x={self.mu_x}, mu_xp={self.mu_xp}, mu_y={self.mu_y}, mu_yp={self.mu_yp}, sigma_x={self.sigma_x}, sigma_xp={self.sigma_xp}, sigma_y={self.sigma_y}, sigma_yp={self.sigma_yp}, sigma_s={self.sigma_s}, sigma_p={self.sigma_p}, energy={self.energy})"
