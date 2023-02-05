import numpy as np
import torch

from cheetah import accelerator as acc

# electron mass in eV
m_e_eV = 510998.8671


def from_astrabeam(path: str):
    """
    Read from a ASTRA beam distribution, and prepare for conversion to a Cheetah
    ParticleBeam or ParameterBeam.

    Adapted from the implementation in ocelot:
    https://github.com/ocelot-collab/ocelot/blob/master/ocelot/adaptors/astra2ocelot.py

    Parameters
    ----------
    path : str
        Path to the ASTRA beam distribution file.

    Returns
    -------
    particles : np.ndarray
        Particle 6D phase space information
    energy : float
        Mean energy of the particle beam

    """
    P0 = np.loadtxt(path)

    # remove lost particles
    inds = np.argwhere(P0[:, 9] > 0)
    inds = inds.reshape(inds.shape[0])

    P0 = P0[inds, :]
    n_particles = P0.shape[0]

    # s_ref = P0[0, 2]
    Pref = P0[0, 5]

    xp = P0[:, :6]
    xp[0, 2] = 0.0
    xp[0, 5] = 0.0

    gamref = np.sqrt((Pref / m_e_eV) ** 2 + 1)
    # energy in eV: E = gamma * m_e
    energy = gamref * m_e_eV

    n_particles = xp.shape[0]
    particles = np.zeros((n_particles, 6))

    u = np.c_[xp[:, 3], xp[:, 4], xp[:, 5] + Pref]
    gamma = np.sqrt(1 + np.sum(u * u, 1) / m_e_eV**2)
    beta = np.sqrt(1 - gamma**-2)
    betaref = np.sqrt(1 - gamref**-2)

    p0 = np.linalg.norm(u, 2, 1).reshape((n_particles, 1))

    u = u / p0
    cdt = -xp[:, 2] / (beta * u[:, 2])
    particles[:, 0] = xp[:, 0] + beta * u[:, 0] * cdt
    particles[:, 2] = xp[:, 1] + beta * u[:, 1] * cdt
    particles[:, 4] = cdt
    particles[:, 1] = xp[:, 3] / Pref
    particles[:, 3] = xp[:, 4] / Pref
    particles[:, 5] = (gamma / gamref - 1) / betaref

    return particles, energy


def ocelot2cheetah(element, warnings=True):
    """
    Translate an Ocelot element to a Cheetah element.

    Parameters
    ----------
    element : ocelot.Element
        Ocelot element object representing an element of particle accelerator.

    Returns
    -------
    cheetah.Element
        Cheetah element object representing an element of particle accelerator.

    Notes
    -----
    Object not supported by Cheetah are translated to drift sections. Screen objects are
    created only from `ocelot.Monitor` objects when the string "SCR" in their `id`
    attribute. Their screen properties are always set to default values and most likely
    need adjusting afterwards. BPM objects are only created from `ocelot.Monitor`
    objects when their id has a substring "BPM".
    """
    try:
        import ocelot as oc
    except ImportError:
        raise ImportError(
            """To use the ocelot2cheetah lattice converter, Ocelot must be first 
        installed, see https://github.com/ocelot-collab/ocelot """
        )

    if isinstance(element, oc.Drift):
        return acc.Drift(element.l, name=element.id)
    elif isinstance(element, oc.Quadrupole):
        return acc.Quadrupole(element.l, element.k1, name=element.id)
    elif isinstance(element, oc.Hcor):
        return acc.HorizontalCorrector(element.l, element.angle, name=element.id)
    elif isinstance(element, oc.Vcor):
        return acc.VerticalCorrector(element.l, element.angle, name=element.id)
    elif isinstance(element, oc.Cavity):
        return acc.Cavity(element.l, name=element.id)
    elif isinstance(element, oc.Monitor) and "BSC" in element.id:
        if warnings:
            print(
                "WARNING: Diagnostic screen was converted with default screen"
                " properties."
            )
        return acc.Screen((2448, 2040), (3.5488e-6, 2.5003e-6), name=element.id)
    elif isinstance(element, oc.Monitor) and "BPM" in element.id:
        return acc.BPM(name=element.id)
    elif isinstance(element, oc.Undulator):
        return acc.Undulator(element.l, name=element.id)
    else:
        if warnings:
            print(
                f"WARNING: Unknown element {element.id}, replacing with drift section."
            )
        return acc.Drift(element.l, name=element.id)


def subcell_of_ocelot(cell, start, end):
    """Extract a subcell `[start, end]` from an Ocelot cell."""
    subcell = []
    is_in_subcell = False
    for el in cell:
        if el.id == start:
            is_in_subcell = True
        if is_in_subcell:
            subcell.append(el)
        if el.id == end:
            break

    return subcell
