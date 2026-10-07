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
    igamma2 = gamma.square().reciprocal()
    beta = (1.0 - igamma2).sqrt()

    return gamma, igamma2, beta


def invert_affine_map(tm: torch.Tensor) -> torch.Tensor:
    """Invert a nonsingular affine map of shape ``(..., 7, 7)``.

    The input transfer map is assumed to be of the form:
        [[ A  b ]
        [ 0  1 ]]
    where A is a 6x6 matrix and b is the 6x1 offset vector.
    This inversion perserves the affine structure of the map compared to a full matrix
    inversion.
    """
    linear_inv = torch.linalg.inv(tm[..., :6, :6])
    offset_inv = -(linear_inv @ tm[..., :6, 6:7])

    inverse = torch.zeros_like(tm)
    inverse[..., :6, :6] = linear_inv
    inverse[..., :6, 6:7] = offset_inv
    inverse[..., 6, 6] = 1.0
    return inverse
