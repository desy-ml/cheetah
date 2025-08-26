import torch


def compute_relativistic_factors(
    energy: torch.Tensor, particle_mass_eV: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the relativistic factors gamma, inverse gamma squared and beta for
    particles.

    :param energy: Energy in eV.
    :param particle_mass_eV: Mass of the particle in eV.
    :return: gamma, igamma2, beta.
    """
    gamma = energy / particle_mass_eV
    igamma2 = (gamma * gamma).reciprocal()  # Division by zero not physical
    beta = (1 - igamma2).sqrt()

    return gamma, igamma2, beta
