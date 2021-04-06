import numpy as np
import ocelot as oc


def generate_particles(n=100000, x=0, y=0, px=0, py=0, sigma_x=175e-9, sigma_y=175e-9,
                       sigma_px=2e-7, sigma_py=2e-7, sigma_s=0, sigma_p=0):
    """Generate n particles according to a multivariate normal distribution."""
    particles = np.ones((int(1e+5), 7))
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
