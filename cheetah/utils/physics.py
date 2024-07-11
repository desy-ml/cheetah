import torch
from scipy.constants import physical_constants

electron_mass_eV = torch.tensor(
    physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
)


def calculate_inverse_gamma_squared(energy):
    gamma = energy / electron_mass_eV.to(energy)
    igamma2 = torch.where(gamma == 0.0, 0.0, 1 / gamma**2)
    return igamma2
