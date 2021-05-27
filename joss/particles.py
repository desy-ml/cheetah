import numpy as np


class Beam:
    """
    Beam of charged particles.

    Parameters
    ----------
    particles : numpy.ndarray
        List of 7-dimensional particle vectors.
    energy : float
        Energy of the beam in eV.
    """

    def __init__(self, particles, energy):
        assert particles.shape[1] == 7, "Particle vectors must be 7-dimensional."
        self.particles = particles
        self.energy = energy
    
    @classmethod
    def make_random(cls, n=100000, mu_x=0, mu_y=0, mu_xp=0, mu_yp=0, sigma_x=175e-9, sigma_y=175e-9,
                    sigma_xp=2e-7, sigma_yp=2e-7, sigma_s=0, sigma_p=0, cor_x=0, cor_y=0, cor_s=0,
                    energy=1e8):
        """
        Generate JOSS Beam of random particles.
        
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
        """
        mean = [mu_x, mu_xp, mu_y, mu_yp, 0, 0]
        cov = [[sigma_x**2,       cor_x,          0,           0,          0,          0],
               [     cor_x, sigma_xp**2,          0,           0,          0,          0],
               [         0,           0, sigma_y**2,       cor_y,          0,          0],
               [         0,           0,      cor_y, sigma_yp**2,          0,          0],
               [         0,           0,          0,           0, sigma_s**2,      cor_s],
               [         0,           0,          0,           0,      cor_s, sigma_p**2]]

        particles = np.ones((n, 7))
        particles[:,:6] = np.random.multivariate_normal(mean, cov, size=n)

        return cls(particles, energy)
    
    
    @classmethod
    def make_linspaced(cls, n=10, mu_x=0, mu_y=0, mu_xp=0, mu_yp=0, sigma_x=175e-9, sigma_y=175e-9,
                       sigma_xp=2e-7, sigma_yp=2e-7, sigma_s=0, sigma_p=0, energy=1e8):
        """
        Generate JOSS Beam of *n* linspaced particles.
        
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
        """
        particles = np.ones((n, 7))
        
        particles[:,0] = np.linspace(mu_x-sigma_x, mu_x+sigma_x, n)
        particles[:,1] = np.linspace(mu_xp-sigma_xp, mu_xp+sigma_xp, n)
        particles[:,2] = np.linspace(mu_y-sigma_y, mu_y+sigma_y, n)
        particles[:,3] = np.linspace(mu_yp-sigma_yp, mu_yp+sigma_yp, n)
        particles[:,4] = np.linspace(-sigma_s, sigma_s, n)
        particles[:,5] = np.linspace(-sigma_p, sigma_p, n)

        return cls(particles, energy)

    @classmethod
    def from_ocelot(cls, parray):
        """
        Convert an Ocelot ParticleArray `parray` to a JOSS Beam.
        """
        print(f"parray = {parray}")
        n = parray.rparticles.shape[1]
        particles = np.ones((n, 7))
        particles[:,:6] = parray.rparticles.transpose()

        return cls(particles, 1e9*parray.E)
    
    @property
    def n(self):
        return self.particles.shape[0]
    
    @property
    def xs(self):
        return self.particles[:,0]
    
    @xs.setter
    def xs(self, value):
        self.particles[:,0] = value
    
    @property
    def mu_x(self):
        return self.xs.mean()
    
    @property
    def sigma_x(self):
        return self.xs.std()
    
    @property
    def xps(self):
        return self.particles[:,1]
    
    @xps.setter
    def xps(self, value):
        self.particles[:,1] = value
    
    @property
    def mu_xp(self):
        return self.xps.mean()
    
    @property
    def sigma_xp(self):
        return self.xps.std()
    
    @property
    def ys(self):
        return self.particles[:,2]
    
    @ys.setter
    def ys(self, value):
        self.particles[:,2] = value
    
    @property
    def mu_y(self):
        return self.ys.mean()
    
    @property
    def sigma_y(self):
        return self.ys.std()
    
    @property
    def yps(self):
        return self.particles[:,3]
    
    @yps.setter
    def yps(self, value):
        self.particles[:,3] = value
    
    @property
    def mu_yp(self):
        return self.yps.mean()
    
    @property
    def sigma_yp(self):
        return self.yps.std()
    
    @property
    def ss(self):
        return self.particles[:,4]
    
    @ss.setter
    def ss(self, value):
        self.particles[:,4] = value
    
    @property
    def mu_s(self):
        return self.ss.mean()
    
    @property
    def sigma_s(self):
        return self.ss.std()
    
    @property
    def ps(self):
        return self.particles[:,5]
    
    @ps.setter
    def ps(self, value):
        self.particles[:,5] = value
    
    @property
    def mu_p(self):
        return self.ps.mean()
    
    @property
    def sigma_p(self):
        return self.ps.std()
    
    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.n}, mu_x={self.mu_x}, mu_xp={self.mu_xp}, mu_y={self.mu_y}, mu_yp={self.mu_yp}, sigma_x={self.sigma_x}, sigma_xp={self.sigma_xp}, sigma_y={self.sigma_y}, sigma_yp={self.sigma_yp}, sigma_s={self.sigma_s}, sigma_p={self.sigma_p}, energy={self.energy})"
