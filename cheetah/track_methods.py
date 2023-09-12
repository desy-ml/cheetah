"""Utility functions for creating transfer maps for the elements."""

from typing import Union

import torch
from scipy import constants

REST_ENERGY = torch.tensor(
    constants.electron_mass
    * constants.speed_of_light**2
    / constants.elementary_charge
)  # electron mass


def rotation_matrix(
    angle: torch.Tensor, device: Union[str, torch.device] = "auto"
) -> torch.Tensor:
    """Rotate the transfer map in x-y plane

    :param angle: Rotation angle in rad, for example `angle = np.pi/2` for vertical =
        dipole.
    :param device: Device used for tracking, by default "auto".
    :return: Rotation matrix to be multiplied to the element's transfer matrix.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cs = torch.cos(angle)
    sn = torch.sin(angle)
    return torch.tensor(
        [
            [cs, 0, sn, 0, 0, 0, 0],
            [0, cs, 0, sn, 0, 0, 0],
            [-sn, 0, cs, 0, 0, 0, 0],
            [0, -sn, 0, cs, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )


def base_rmatrix(
    length: torch.Tensor,
    k1: torch.Tensor,
    hx: torch.Tensor,
    tilt: torch.Tensor = torch.tensor(0.0),
    energy: torch.Tensor = torch.tensor(0.0),
    device: Union[str, torch.device] = "auto",
) -> torch.Tensor:
    """
    Create a universal transfer matrix for a beamline element.

    :param length: Length of the element in m.
    :param k1: Quadrupole strength in 1/m**2.
    :param hx: Curvature (1/radius) of the element in 1/m**2.
    :param tilt: Roation of the element relative to the longitudinal axis in rad.
    :param energy: Beam energy in eV.
    :param device: Device where the transfer matrix is created. If "auto", the device
        is selected automatically.
    :return: Transfer matrix for the element.
    """

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    gamma = energy / REST_ENERGY
    igamma2 = 1 / gamma**2 if gamma != 0 else torch.tensor(0.0)

    beta = torch.sqrt(1 - igamma2)

    kx2 = k1 + hx**2
    ky2 = -k1
    kx = torch.sqrt(kx2 + 0.0j)
    ky = torch.sqrt(ky2 + 0.0j)
    cx = torch.cos(kx * length).real
    cy = torch.cos(ky * length).real
    sy = (torch.sin(ky * length) / ky).real if ky != 0 else length

    if kx != 0:
        sx = (torch.sin(kx * length) / kx).real
        dx = hx / kx2 * (1.0 - cx)
        r56 = hx**2 * (length - sx) / kx2 / beta**2
    else:
        sx = length
        dx = length**2 * hx / 2
        r56 = hx**2 * length**3 / 6 / beta**2

    r56 -= length / beta**2 * igamma2

    R = torch.tensor(
        [
            [cx, sx, 0, 0, 0, dx / beta, 0],
            [-kx2 * sx, cx, 0, 0, 0, sx * hx / beta, 0],
            [0, 0, cy, sy, 0, 0, 0],
            [0, 0, -ky2 * sy, cy, 0, 0, 0],
            [sx * hx / beta, dx / beta, 0, 0, 1, r56, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )

    # Rotate the R matrix for skew / vertical magnets
    if tilt != 0:
        R = torch.matmul(torch.matmul(rotation_matrix(-tilt), R), rotation_matrix(tilt))
    return R


def misalignment_matrix(
    misalignment: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift the beam for tracking beam through misaligned elements"""
    R_exit = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, misalignment[0]],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, misalignment[1]],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )
    R_entry = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, -misalignment[0]],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, -misalignment[1]],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )
    return R_exit, R_entry
