import numpy as np
import ocelot as oc


class Beam:

    def __init__(self, particles):
        self.particles = particles
    
    @classmethod
    def make_random(cls, n=100000, x=0, y=0, xp=0, yp=0, sigma_x=175e-9, sigma_y=175e-9,
                    sigma_xp=2e-7, sigma_yp=2e-7, sigma_s=0, sigma_p=0):
        """
        Generate JOSS Beam of random particles.
        
        Parameters
        ----------
        n : int, optional
            Number of particles to generate.
        x : float, optional
            Center of the particle distribution on x in meters.
        y : float, optional
            Center of the particle distribution on y in meters.
        xp : float, optional
            Center of the particle distribution on px in meters.
        yp : float, optional
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
            CURRENTLY NOT IN USE! Sgima of the particle distribution in s direction in meters.
        sigma_p : float, optional
            Sgima of the particle distribution in p direction in meters.
        """
        parray = oc.generate_parray(nparticles=n,
                                    sigma_x=sigma_x,
                                    sigma_xp=sigma_xp,
                                    sigma_y=sigma_y,
                                    sigma_yp=sigma_yp,
                                    sigma_p=sigma_p,
                                    chirp=0.0,
                                    energy=0.1,
                                    sigma_tau=0.0)
        beam = cls.from_ocelot(parray)

        beam.particles[:,0] += x
        beam.particles[:,1] += xp
        beam.particles[:,2] += y
        beam.particles[:,3] += yp

        return beam
    
    @classmethod
    def make_linspaced(cls, n=10, mu_x=0, mu_y=0, mu_xp=0, mu_yp=0, sigma_x=175e-9, sigma_y=175e-9,
                       sigma_xp=2e-7, sigma_yp=2e-7, sigma_s=0, sigma_p=0):
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

        return cls(particles)

    @classmethod
    def from_ocelot(cls, parray):
        """
        Convert an Ocelot ParticleArray `parray` to a JOSS Beam.
        """
        n = parray.rparticles.shape[1]
        particles = np.ones((n, 7))
        particles[:,:6] = parray.rparticles.transpose()

        return cls(particles)
    
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
