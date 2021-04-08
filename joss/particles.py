import numpy as np
import ocelot as oc


def generate_particles(n=100000, x=0, y=0, px=0, py=0, sigma_x=175e-9, sigma_y=175e-9,
                       sigma_px=2e-7, sigma_py=2e-7, sigma_s=0, sigma_p=0):
    """
    Generate n particles using Ocelot's particle generation.
    
    Parameters
    ----------
    n : int, optional
        Number of particles to generate.
    x : float, optional
        CURRENTLY NOT IN USE! Center of the particle distribution on x in meters.
    y : float, optional
        CURRENTLY NOT IN USE! Center of the particle distribution on y in meters.
    px : float, optional
        CURRENTLY NOT IN USE! Center of the particle distribution on px in meters.
    py : float, optional
        CURRENTLY NOT IN USE! Center of the particle distribution on py in meters.
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
    
    Returns
    -------
    numpy.ndarray
        Randomly generated particles.
    """
    particles = np.ones((n, 7))
    particles[:,:6] = oc.generate_parray(nparticles=n,
                                         sigma_x=sigma_x,
                                         sigma_px=sigma_px,
                                         sigma_y=sigma_y,
                                         sigma_py=sigma_py,
                                         sigma_p=sigma_p,
                                         chirp=0,
                                         energy=0.1,
                                         sigma_tau=0.0).rparticles.transpose()
    return particles
