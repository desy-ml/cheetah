import numpy as np


def generate_particles(n=100000, x=0, y=0, px=0, py=0, sigma_x=175e-9, sigma_y=175e-9,
                       sigma_px=2e-7, sigma_py=2e-7, sigma_s=0, sigma_p=0):
    """Generate n particles according to a multivariate normal distribution."""
    mean = [x, y, px, py, 0, 0]
    rx = 0
    ry = 0
    cov = [[sigma_x**2,          rx,          0,           0,          0,          0],
           [        rx, sigma_px**2,          0,           0,          0,          0],
           [         0,           0, sigma_y**2,          ry,          0,          0],
           [         0,           0,         ry, sigma_py**2,          0,          0],
           [         0,           0,          0,           0, sigma_s**2,          0],
           [         0,           0,          0,           0,          0, sigma_p**2]]

    particles = np.random.multivariate_normal(mean=mean, cov=cov, size=n)
    return particles


def track(particles, transfer_map):
    """Track particles through a given transfer map."""
    print(f"Multiply {particles.shape[0]}x{particles.shape[1]} with {transfer_map.shape[0]}x{transfer_map.shape[1]}")
    return np.matmul(particles, transfer_map.transpose())
    # return np.matmul(transfer_map, particles.transpose()).transpose()
