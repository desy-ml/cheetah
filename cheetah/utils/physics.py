import torch
from scipy.constants import physical_constants

electron_mass_eV = torch.tensor(
    physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
)


def compute_relativistic_factors(energy):
    """
    calculates relativistic factors gamma, inverse gamma squared, beta
    for electrons

    :param energy: Energy in eV
    :return: gamma, igamma2, beta
    """
    gamma = energy / electron_mass_eV.to(energy)
    igamma2 = torch.where(gamma == 0.0, 0.0, 1 / gamma**2)
    beta = torch.sqrt(1 - igamma2)
    return gamma, igamma2, beta
