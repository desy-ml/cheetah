import numpy as np
from scipy.constants import physical_constants

# Electron mass in eV
electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


def from_astrabeam(path: str) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Read from a ASTRA beam distribution, and prepare for conversion to a Cheetah
    ParticleBeam or ParameterBeam.

    Adapted from the implementation in Ocelot:
    https://github.com/ocelot-collab/ocelot/blob/master/ocelot/adaptors/astra2ocelot.py

    :param path: Path to the ASTRA beam distribution file.
    :return: (particles, energy, q_array)
        Particle 6D phase space information, mean energy,
        and the charge array of the particle beam.
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

    gamref = np.sqrt((Pref / electron_mass_eV) ** 2 + 1)
    # energy in eV: E = gamma * m_e
    energy = gamref * electron_mass_eV

    n_particles = xp.shape[0]
    particles = np.zeros((n_particles, 6))

    u = np.c_[xp[:, 3], xp[:, 4], xp[:, 5] + Pref]
    gamma = np.sqrt(1 + np.sum(u * u, 1) / electron_mass_eV**2)
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

    q_array = abs(P0[:, 7]) * 1e-9  # convert charge array from nC to C

    return particles, energy, q_array
