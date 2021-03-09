import numpy as np
import ocelot as oc


def ocelot_lattice_2_transfer_matrix(lattice):
    """Compute the transfer matrix of an Ocelot lattice."""
    transfer_map = np.eye(6)
    for element in lattice:
        if element.__class__ is oc.Drift:
            transfer_map = np.matmul(drift(element.l), transfer_map)
            # transfer_map = np.matmul(quadrupole(element.l, 0.0), transfer_map)
        elif element.__class__ is oc.Quadrupole:
            transfer_map = np.matmul(quadrupole(element.l, element.k1), transfer_map)
        else:
            transfer_map = np.matmul(drift(element.l), transfer_map)
        
    return transfer_map


def drift(l):
    """Create the transfer matrix of a drift section of given length."""
    
    m_e_kg = 9.10938215e-31      # kg
    speed_of_light = 299792458.0 # m/s
    q_e = 1.6021766208e-19       # C - Elementary charge
    m_e_eV = m_e_kg * speed_of_light**2 / q_e  # eV (510998.8671)
    m_e_GeV = m_e_eV / 1e+9                    # GeV

    energy = 0.1    # GeV
    gamma = energy / m_e_GeV
    
    igamma2 = 1. / gamma**2 if gamma != 0 else 0

    transfer_map = np.array([[1, l, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1, l, 0, 0],
                             [0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, l*igamma2],
                             [0, 0, 0, 0, 0, 1]])
    return transfer_map


def quadrupole(l, k1):
    """Create the transfer matrix of a quadrupole magnet of the given parameters."""
    
    m_e_kg = 9.10938215e-31      # kg
    speed_of_light = 299792458.0 # m/s
    q_e = 1.6021766208e-19       # C - Elementary charge
    m_e_eV = m_e_kg * speed_of_light**2 / q_e  # eV (510998.8671)
    m_e_GeV = m_e_eV / 1e+9  

    energy = 0.1    # GeV
    gamma = energy / m_e_GeV

    igamma2 = 0.

    if gamma != 0:
        igamma2 = 1. / (gamma * gamma)

    beta = np.sqrt(1. - igamma2)
    
    hx = 0
    kx2 = (k1+hx*hx)
    ky2 = -k1
    kx = np.sqrt(kx2 + 0.j)
    ky = np.sqrt(ky2 + 0.j)
    cx = np.cos(kx * l).real
    cy = np.cos(ky * l).real
    sy = (np.sin(ky * l) / ky).real if ky != 0 else l

    if kx != 0:
        sx = (np.sin(kx * l) / kx).real
        dx = hx / kx2 * (1. - cx)
        r56 = hx * hx * (l - sx) / kx2 / beta ** 2
    else:
        sx = l
        dx = l * l * hx / 2.
        r56 = hx * hx * l ** 3 / 6. / beta ** 2
    
    r56 -= l / (beta * beta) * igamma2

    transfer_map = np.array([[            cx,        sx,        0., 0., 0.,      dx / beta],
                             [-kx2 * sx,             cx,        0., 0., 0., sx * hx / beta],
                             [            0.,        0.,        cy, sy, 0.,             0.],
                             [            0.,        0., -ky2 * sy, cy, 0.,             0.],
                             [sx * hx / beta, dx / beta,        0., 0., 1.,            r56],
                             [            0.,        0.,        0., 0., 0.,             1.]])

    return transfer_map
