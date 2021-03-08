import numpy as np


def generate_particles(n=100000, x=0, y=0, px=0, py=0, sigma_x=175e-9, sigma_y=175e-9,
                       sigma_px=2e-7, sigma_py=2e-7):
    """Generate n particles according to a multivariate normal distribution."""
    mean = [x, y, px, py]
    rx = 0
    ry = 0
    cov = [[sigma_x**2, rx, 0, 0],
           [rx, sigma_px**2, 0, 0],
           [0, 0, sigma_y**2, ry],
           [0, 0, ry, sigma_py**2]]

    particles = np.random.multivariate_normal(mean=mean, cov=cov, size=n)
    return particles


def track(particles, transfer_map):
    """Track particles through a given transfer map."""
    return np.dot(particles, transfer_map)
