import math
import torch


CLASSICAL_PROTON_RADIUS = 1.53469e-18


def get_lorentz_factors(rest_energy: float, kin_energy: float) -> tuple[float, float]:
    gamma = 1.0 + (kin_energy / rest_energy)
    beta = math.sqrt(1.0 - (1.0 / gamma)**2)
    return (gamma, beta)


def get_perveance(rest_energy: float, kin_energy: float, line_density: float) -> float:
    gamma, beta = get_lorentz_factors(rest_energy, kin_energy)

    classical_proton_radius = CLASSICAL_PROTON_RADIUS
    perveance = (2.0 * classical_proton_radius * line_density) / (beta**2 * gamma**3)

    return perveance


def build_norm_matrix_from_twiss_2d(alpha: float, beta: float, eps: float = None) -> torch.Tensor:
    alpha = torch.as_tensor(alpha)
    beta = torch.as_tensor(beta)

    V = torch.tensor([[beta, 0.0], [-alpha, 1.0]]) * math.sqrt(1.0 / beta)

    if eps is not None:
        eps = torch.as_tensor(eps)
        A = torch.eye(2) * torch.sqrt(eps)
        V = torch.matmul(V, A)

    return torch.linalg.inv(V)


def build_rotation_matrix(angle: float) -> torch.Tensor:
    matrix = torch.zeros((2, 2))
    matrix[0, 0] = +math.cos(angle)
    matrix[0, 1] = +math.sin(angle)
    matrix[1, 0] = -math.sin(angle)
    matrix[1, 1] = +math.cos(angle)
    return matrix

