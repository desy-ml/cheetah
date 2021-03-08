from copy import Error
import numpy as np
import ocelot as oc

from joss import utils


def ocelot_lattice_2_transfer_matrix(lattice):
    """Compute the transfer matrix of an Ocelot lattice."""
    transfer_map = np.eye(4)
    for element in lattice:
        if element.__class__ is oc.Drift:
            transfer_map = np.matmul(drift(element.l), transfer_map)
        elif element.__class__ is oc.Quadrupole:
            transfer_map = np.matmul(quadrupole(element.l, element.k1))
        else:
            raise Exception(f"Elements of type {element.__class__} are not yet supported by JOSS")
    
    return transfer_map


def drift(l):
    """Create the transfer matrix of a drift section of given length."""
    transfer_map = np.array([[1, l, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, l],
                             [0, 0, 0, 1]])
    return transfer_map


def quadrupole(l, k1):
    """Create the transfer matrix of a quadrupole magnet of the given parameters."""
    if k1 > 0:
        transfer_map = np.array([[np.cos(np.sqrt(k1) * l), np.sin(np.sqrt(k1) * l) / np.sqrt(k1), 0, 0],
                                 [-np.sqrt(k1) * np.cos(np.sqrt(k1) * l), np.cos(np.sqrt(k1) * l), 0, 0],
                                 [0, 0, np.cosh(np.sqrt(k1) * l), np.sinh(np.sqrt(k1) * l) / np.sqrt(k1)],
                                 [0, 0, np.sqrt(k1) * np.sinh(np.sqrt(k1) * l), np.cosh(np.sqrt(k1) * l)]])
    elif k1 < 0:
        k1 = abs(k1)
        transfer_map = np.array([[np.cosh(np.sqrt(k1) * l), np.sinh(np.sqrt(k1) * l) / np.sqrt(k1), 0, 0],
                                 [np.sqrt(k1) * np.cosh(np.sqrt(k1) * l), np.cosh(np.sqrt(k1) * l), 0, 0],
                                 [0, 0, np.cos(np.sqrt(k1) * l), np.sin(np.sqrt(k1) * l) / np.sqrt(k1)],
                                 [0, 0, -np.sqrt(k1) * np.sin(np.sqrt(k1) * l), np.cos(np.sqrt(k1) * l)]])
    elif k1 == 0:
        transfer_map = np.array([[1, l, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, l],
                                 [0, 0, 0, 1]])
    return transfer_map
