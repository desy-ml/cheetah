import numpy as np
import ocelot as oc
from scipy import constants


REST_ENERGY = constants.electron_mass * constants.speed_of_light**2 / constants.elementary_charge


def ocelot_lattice_2_transfer_matrix(lattice):
    """Compute the transfer matrix of an Ocelot lattice."""
    transfer_map = np.eye(6)
    for element in lattice:
        if element.__class__ is oc.Drift:
            transfer_map = np.matmul(drift(element.l), transfer_map)
        elif element.__class__ is oc.Quadrupole:
            transfer_map = np.matmul(quadrupole(element.l, element.k1), transfer_map)
        else:
            transfer_map = np.matmul(drift(element.l), transfer_map)
        
    return transfer_map


def drift(l, energy=1e+8):
    """Create the transfer matrix of a drift section of given length."""
    gamma = energy / REST_ENERGY
    igamma2 = 1 / gamma**2 if gamma != 0 else 0

    transfer_map = np.array([[1, l, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1, l, 0, 0],
                             [0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, l*igamma2],
                             [0, 0, 0, 0, 0, 1]])
    return transfer_map


def quadrupole(l, k1, energy=1e+8):
    """Create the transfer matrix of a quadrupole magnet of the given parameters."""
    gamma = energy / REST_ENERGY
    igamma2 = 1 / gamma**2 if gamma != 0 else 0

    beta = np.sqrt(1 - igamma2)
    
    hx = 0
    kx2 = k1 + hx**2
    ky2 = -k1
    kx = np.sqrt(kx2 + 0.j)
    ky = np.sqrt(ky2 + 0.j)
    cx = np.cos(kx * l).real
    cy = np.cos(ky * l).real
    sy = (np.sin(ky * l) / ky).real if ky != 0 else l

    if kx != 0:
        sx = (np.sin(kx * l) / kx).real
        dx = hx / kx2 * (1. - cx)
        r56 = hx**2 * (l - sx) / kx2 / beta**2
    else:
        sx = l
        dx = l**2 * hx / 2
        r56 = hx**2 * l**3 / 6 / beta**2
    
    r56 -= l / beta**2 * igamma2

    transfer_map = np.array([[            cx,        sx,        0., 0., 0.,      dx / beta],
                             [     -kx2 * sx,        cx,        0., 0., 0., sx * hx / beta],
                             [            0.,        0.,        cy, sy, 0.,             0.],
                             [            0.,        0., -ky2 * sy, cy, 0.,             0.],
                             [sx * hx / beta, dx / beta,        0., 0., 1.,            r56],
                             [            0.,        0.,        0., 0., 0.,             1.]])

    return transfer_map
