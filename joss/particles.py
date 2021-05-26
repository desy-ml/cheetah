import numpy as np
import ocelot as oc


class Beam:

    def __init__(self, particles):
        self.particles = particles
    
    @classmethod
    def make_random(cls, n=100000, x=0, y=0, px=0, py=0, sigma_x=175e-9, sigma_y=175e-9,
                    sigma_px=2e-7, sigma_py=2e-7, sigma_s=0, sigma_p=0):
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
        px : float, optional
            Center of the particle distribution on px in meters.
        py : float, optional
            Center of the particle distribution on py in meters.
        sigma_x : float, optional
            Sgima of the particle distribution in x direction in meters.
        sigma_y : float, optional
            Sgima of the particle distribution in y direction in meters.
        sigma_px : float, optional
            Sgima of the particle distribution in px direction in meters.
        sigma_py : float, optional
            Sgima of the particle distribution in py direction in meters.
        sigma_s : float, optional
            CURRENTLY NOT IN USE! Sgima of the particle distribution in s direction in meters.
        sigma_p : float, optional
            Sgima of the particle distribution in p direction in meters.
        """
        parray = oc.generate_parray(nparticles=n,
                                    sigma_x=sigma_x,
                                    sigma_px=sigma_px,
                                    sigma_y=sigma_y,
                                    sigma_py=sigma_py,
                                    sigma_p=sigma_p,
                                    chirp=0,
                                    energy=0.1,
                                    sigma_tau=0.0)
        beam = cls.from_ocelot(parray)

        beam.particles[:,0] += x
        beam.particles[:,1] += px
        beam.particles[:,2] += y
        beam.particles[:,3] += py

        return beam
    
    @classmethod
    def make_linspaced(cls, n=10, x=0, y=0, px=0, py=0, sigma_x=175e-9, sigma_y=175e-9,
                       sigma_px=2e-7, sigma_py=2e-7, sigma_s=0, sigma_p=0):
        """
        Generate JOSS Beam of *n* linspaced particles.
        
        Parameters
        ----------
        n : int, optional
            Number of particles to generate.
        x : float, optional
            Center of the particle distribution on x in meters.
        y : float, optional
            Center of the particle distribution on y in meters.
        px : float, optional
            Center of the particle distribution on px in meters.
        py : float, optional
            Center of the particle distribution on py in meters.
        sigma_x : float, optional
            Sgima of the particle distribution in x direction in meters.
        sigma_y : float, optional
            Sgima of the particle distribution in y direction in meters.
        sigma_px : float, optional
            Sgima of the particle distribution in px direction in meters.
        sigma_py : float, optional
            Sgima of the particle distribution in py direction in meters.
        sigma_s : float, optional
            Sgima of the particle distribution in s direction in meters.
        sigma_p : float, optional
            Sgima of the particle distribution in p direction in meters.
        """
        particles = np.ones((n, 7))
        
        particles[:,0] = np.linspace(x-sigma_x, x+sigma_x, n)
        particles[:,1] = np.linspace(px-sigma_px, px+sigma_px, n)
        particles[:,2] = np.linspace(y-sigma_y, y+sigma_y, n)
        particles[:,3] = np.linspace(py-sigma_py, py+sigma_py, n)
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
    def x(self):
        return self.particles[:,0]
    
    @x.setter
    def x(self, value):
        self.particles[:,0] = value
    
    @property
    def px(self):
        return self.particles[:,1]
    
    @px.setter
    def px(self, value):
        self.particles[:,1] = value
    
    @property
    def y(self):
        return self.particles[:,2]
    
    @y.setter
    def y(self, value):
        self.particles[:,2] = value
    
    @property
    def py(self):
        return self.particles[:,3]
    
    @py.setter
    def py(self, value):
        self.particles[:,3] = value
    
    @property
    def s(self):
        return self.particles[:,4]
    
    @s.setter
    def s(self, value):
        self.particles[:,4] = value
    
    @property
    def p(self):
        return self.particles[:,5]
    
    @p.setter
    def p(self, value):
        self.particles[:,5] = value
