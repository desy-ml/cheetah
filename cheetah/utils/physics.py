import torch
from scipy.constants import physical_constants

electron_mass_eV = physical_constants["electron mass energy equivalent in MeV"][0] * 1e6


def compute_relativistic_factors(
    energy: torch.Tensor,
    particle_mass_eV: float = electron_mass_eV,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the relativistic factors gamma, inverse gamma squared and beta for
    electrons.

    :param energy: Energy in eV.
    :return: gamma, igamma2, beta.
    """
    gamma = energy / particle_mass_eV
    igamma2 = torch.where(gamma == 0.0, 0.0, 1 / gamma**2)
    beta = torch.sqrt(1 - igamma2)

    return gamma, igamma2, beta
